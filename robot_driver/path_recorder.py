import threading
import time
import json
from typing import List, Dict, Optional

from arm_driver import RobotArmDriver
from robot_logger import RobotLogger

class PathRecorder:
    """
    Records and manages robot movement paths.
    
    This class captures the robot's state (joint positions) at regular
    intervals and can save this data to a file, or load a path from a file.
    """
    
    def __init__(self, arm_driver: RobotArmDriver, logger: Optional[RobotLogger] = None):
        """
        Initialize the PathRecorder.
        
        Args:
            arm_driver: An instance of the RobotArmDriver to get status from.
            logger: An optional logger instance. If None, a new one is created.
        """
        self.arm_driver = arm_driver
        self.logger = logger or RobotLogger("PathRecorder")
        
        self.is_recording = False
        self.recorded_path: List[Dict] = []
        self._recording_thread: Optional[threading.Thread] = None
        
        self.logger.info("PathRecorder initialized.")

    def start_recording(self, interval: float = 0.1):
        """
        Starts recording the robot's path in a background thread.
        
        Args:
            interval (float): The time interval (in seconds) between position captures.
        """
        if self.is_recording:
            self.logger.warning("Recording is already in progress.")
            return
            
        if not self.arm_driver.is_connected()[1]: # Check data connection
            self.logger.error("Cannot start recording. Robot data port is not connected.")
            return

        self.is_recording = True
        self.recorded_path = []  # Clear any previous path
        
        self._recording_thread = threading.Thread(
            target=self._recording_loop, 
            args=(interval,), 
            daemon=True
        )
        self._recording_thread.start()
        self.logger.info(f"Path recording started with a {interval}s interval.")

    def stop_recording(self):
        """
        Stops the current recording session.
        """
        if not self.is_recording:
            self.logger.warning("Recording is not currently active.")
            return
            
        self.is_recording = False
        if self._recording_thread and self._recording_thread.is_alive():
            self._recording_thread.join()  # Wait for the thread to finish
        
        self.logger.info(f"Path recording stopped. Captured {len(self.recorded_path)} points.")

    def _recording_loop(self, interval: float):
        """
        The internal loop that runs in a separate thread to capture path points.
        
        Args:
            interval (float): The sleep time between captures.
        """
        while self.is_recording:
            try:
                # Get the latest system status from the driver
                status = self.arm_driver.sys_status
                
                # Create a data point for the path with only timestamp and joints
                path_point = {
                    "timestamp": time.time(),
                    "jnt_ang": status.jnt_ang,
                }
                
                self.recorded_path.append(path_point)
                
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error during recording loop: {e}")
                self.is_recording = False # Stop recording on error

    def save_path(self, filepath: str) -> bool:
        """
        Saves the recorded path to a JSON file.
        
        Args:
            filepath (str): The path to the file where the path will be saved.
            
        Returns:
            bool: True if saving was successful, False otherwise.
        """
        if not self.recorded_path:
            self.logger.warning("No path data to save.")
            return False
            
        try:
            with open(filepath, 'w') as f:
                json.dump(self.recorded_path, f, indent=4)
            self.logger.info(f"Path successfully saved to {filepath}")
            return True
        except IOError as e:
            self.logger.error(f"Failed to save path to {filepath}: {e}")
            return False

    def load_path(self, filepath: str) -> Optional[List[Dict]]:
        """
        Loads a robot path from a JSON file.
        
        This will overwrite any currently recorded path data.
        
        Args:
            filepath (str): The path to the JSON file to load.
            
        Returns:
            The loaded path data, or None if loading fails.
        """
        try:
            with open(filepath, 'r') as f:
                self.recorded_path = json.load(f)
            self.logger.info(f"Path successfully loaded from {filepath}. Contains {len(self.recorded_path)} points.")
            return self.recorded_path
        except FileNotFoundError:
            self.logger.error(f"Path file not found: {filepath}")
            return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to decode JSON from {filepath}: {e}")
            return None
        except IOError as e:
            self.logger.error(f"Failed to read path file {filepath}: {e}")
            return None

    def get_path(self) -> List[Dict]:
        """
        Returns the currently stored path data.
        
        Returns:
            A list of path point dictionaries.
        """
        return self.recorded_path

# ============================================================================
# Main Execution Block with Upfront Filename Input
# ============================================================================
if __name__ == "__main__":
    
    # Change this IP address to your robot's actual IP address.
    robot_ip_address = "192.168.0.10"
    
    # Initialize the robot driver and path recorder
    robot = RobotArmDriver(robot_ip=robot_ip_address) 
    recorder = PathRecorder(arm_driver=robot)

    # Start the robot driver
    if not robot.start():
        print(f"Failed to start robot driver. Check connection to {robot_ip_address}. Exiting.")
        exit()
        
    print(f"Robot driver started. Waiting for connections to {robot_ip_address}...")
    
    # Wait for both command and data connections to be established
    while not (robot.is_connected()[0] and robot.is_connected()[1]):
        time.sleep(0.5)
    
    print("Robot is connected.")
    
    # Check and handle robot initialization
    print("\nChecking robot state...")
    time.sleep(1)  # Allow time for status update
    
    # Check if robot needs initialization
    if robot.sys_status.init_state_info == 0:
        print("Robot not initialized. Initializing...")
        robot.robot_init()
        time.sleep(3)  # Wait for initialization
        
        # Verify initialization succeeded
        if robot.sys_status.init_error != 0:
            print("Robot initialization failed. Please check the robot. Exiting.")
            robot.stop()
            exit()
    
    # Check if robot is in simulation mode and switch to real mode
    if robot.sys_status.program_mode == 1:
        print("Robot is in simulation mode. Switching to real mode...")
        robot.set_program_mode_real()
        time.sleep(1)  # Wait for mode change
    
    print("Robot is ready.")

    # Step 1: Get the filename upfront
    filename = input("\nEnter the filename for the recording (e.g., my_path.json): ")
    if not filename.endswith('.json'):
        filename += '.json'

    try:
        is_recording_started = False
        while True:
            print("\n" + "="*30)
            
            if not is_recording_started:
                # Menu before recording starts
                print("  Robot Path Recorder Menu")
                print("="*30)
                print("1. Start Recording")
                print("2. Exit")
                print("="*30)
                choice = input("Enter your choice (1-2): ")

                if choice == '1':
                    recorder.start_recording(interval=0.1) # Start recording with 0.1s interval or 10Hz
                    is_recording_started = True
                elif choice == '2':
                    print("Exiting application.")
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            
            else:
                # Menu while recording is active
                print("  RECORDING IN PROGRESS...")
                print("="*30)
                print("1. Stop and Save Path")
                print("="*30)
                choice = input("Enter your choice (1): ")

                if choice == '1':
                    recorder.stop_recording()
                    
                    if recorder.save_path(filename):
                        print(f"Path saved to {filename}")
                    else:
                        print("Failed to save path.")
                    
                    print("Exiting application.")
                    break
                else:
                    print("Invalid choice. Please enter 1.")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Exiting.")
        
    finally:
        # Cleanly stop the robot driver
        print("\n--- Stopping robot driver. ---")
        robot.stop()
        print("Application finished.")