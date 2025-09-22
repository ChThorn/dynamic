"""
Blend Motion Methods:
- move_joint_blend_clear() - Clear joint blend points
- move_joint_blend_add_point() - Add joint blend point
- move_joint_blend_move_point() - Execute joint blend motion
- move_tcp_blend_clear() - Clear TCP blend points  
- move_tcp_blend_add_point() - Add TCP blend point
- move_tcp_blend_move_point() - Execute TCP blend motion

Circle Motion Methods:
- move_circle_three_point() - Circular motion using three points
- move_circle_axis() - Circular motion around axis

"""

import socket
import struct
import threading
import time
from typing import Optional, Callable, List, Tuple
# --- IMPORT ---
from robot_data import *
from robot_logger import RobotLogger
from tcp_server import RobotTCPServer

class RobotArmDriver:
    
    
    def __init__(self, robot_ip="192.168.0.10", cmd_port=5000, data_port=5001, 
                 motion_server_port=7000, logger=None):
        """Initialize robot arm driver"""
        self.robot_ip = robot_ip
        self.cmd_port = cmd_port
        self.data_port = data_port
        self.motion_server_port = motion_server_port
        
        self.logger = logger or RobotLogger("RobotArmDriver")
        
        # Socket connections
        self.cmd_socket = None
        self.data_socket = None
        
        # Connection status
        self.cmd_connection_status = False
        self.data_connection_status = False
        self.robot_moving = False

        # Better blend motion tracking
        self.is_blended_move_active = False
        self.previous_robot_state = RobotState.IDLE.value
        
        # Motion server
        self.motion_server = RobotTCPServer(port=motion_server_port, logger=self.logger)
        self.motion_server.set_connection_callback(self._on_motion_server_connected)
        self.motion_server.set_disconnection_callback(self._on_motion_server_disconnected)
        self.motion_server.set_data_received_callback(self._on_motion_server_data)
        
        # Robot status data
        self.sys_status = SystemStatus()
        self.sys_config = SystemConfig()
        self.sys_popup = SystemPopup()
        
        # Control flags
        self.cmd_confirm_flag = False
        self.move_cmd_flag = False
        self.move_cmd_cnt = 0
        self.system_forced_stop_flag = False
        self.command_out_flag = False
        self.data_recv_count = 0
        self.ai1_connected = False
        
        # Error flags
        self.fatal_info_robot_connection_error = False
        self.fatal_info_robot_miss_command_working_check = False
        self.fatal_info_robot_data_error = False # Flag for data reception timeout
        self.super_fatal_error_robot_miss_command_working_check = False
        self.super_fatal_error_robot_data = False
        self.debug_miss_command_working_check_count = 0
        
        # Data buffer
        self.recv_buffer = bytearray()
        
        # Threading
        self.update_thread = None
        self.running = False
        self.update_interval = 0.1  # 100ms
        
        # Callbacks
        self.status_update_callback: Optional[Callable] = None
        self.connection_status_callback: Optional[Callable] = None
        self.error_callback: Optional[Callable] = None
    
    # Callback setters
    def set_status_update_callback(self, callback: Callable):
        """Set callback for status updates"""
        self.status_update_callback = callback
    
    def set_connection_status_callback(self, callback: Callable):
        """Set callback for connection status changes"""
        self.connection_status_callback = callback
    
    def set_error_callback(self, callback: Callable):
        """Set callback for error notifications"""
        self.error_callback = callback
    
    # Lifecycle methods
    def start(self) -> bool:
        """Start the robot driver"""
        try:
            # Start motion server
            if not self.motion_server.start_server():
                self.logger.error("Failed to start motion server")
                return False
            
            # Start update thread
            self.running = True
            self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self.update_thread.start()
            
            self.logger.info("Robot arm driver started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start robot driver: {e}")
            return False
    
    def stop(self):
        """Stop the robot driver"""
        if not self.running:
            return
            
        self.logger.info("Stopping robot arm driver...")
        self.running = False
        
        # Wait for the update thread to cleanly terminate before closing resources
        if self.update_thread and self.update_thread.is_alive():
            self.logger.debug("Waiting for update thread to terminate...")
            self.update_thread.join(timeout=3.0)
            if self.update_thread.is_alive():
                self.logger.warning("Update thread did not terminate cleanly")

        # Stop motion server and ensure it's fully stopped
        self.logger.debug("Stopping motion server...")
        self.motion_server.stop_server()
        
        # Close socket connections
        self.logger.debug("Closing socket connections...")
        self.disconnect_cmd()
        self.disconnect_data()
        
        # Small delay to ensure all resources are released
        import time
        time.sleep(0.2)
        
        self.logger.info("Robot arm driver stopped")
    
    # Connection management
    def connect_cmd(self) -> bool:
        """Connect to command port"""
        if self.cmd_connection_status:
            return True
        
        try:
            self.cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.cmd_socket.settimeout(5.0)
            self.cmd_socket.connect((self.robot_ip, self.cmd_port))
            self.cmd_connection_status = True
            
            self.logger.info(f"Connected to command port {self.robot_ip}:{self.cmd_port}")
            
            if self.connection_status_callback:
                self.connection_status_callback("command", True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to command port: {e}")
            self.cmd_connection_status = False
            self.cmd_socket = None
            return False
    
    def connect_data(self) -> bool:
        """Connect to data port"""
        if self.data_connection_status:
            return True
        
        try:
            self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.data_socket.settimeout(5.0)
            self.data_socket.connect((self.robot_ip, self.data_port))
            self.data_connection_status = True
            
            self.logger.info(f"Connected to data port {self.robot_ip}:{self.data_port}")
            
            if self.connection_status_callback:
                self.connection_status_callback("data", True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to data port: {e}")
            self.data_connection_status = False
            self.data_socket = None
            return False
    
    def disconnect_cmd(self):
        """Disconnect from command port"""
        if self.cmd_socket:
            try:
                self.cmd_socket.close()
            except:
                pass
            self.cmd_socket = None
        
        if self.cmd_connection_status:
            self.cmd_connection_status = False
            self.logger.info("Disconnected from command port")
            
            if self.connection_status_callback:
                self.connection_status_callback("command", False)
    
    def disconnect_data(self):
        """Disconnect from data port"""
        if self.data_socket:
            try:
                self.data_socket.close()
            except:
                pass
            self.data_socket = None
        
        if self.data_connection_status:
            self.data_connection_status = False
            self.logger.info("Disconnected from data port")
            
            if self.connection_status_callback:
                self.connection_status_callback("data", False)
    
    # Status methods
    def is_connected(self) -> Tuple[bool, bool]:
        """Get connection status"""
        return self.cmd_connection_status, self.data_connection_status
    
    def is_motion_idle(self) -> bool:
        """
        Check if robot motion is idle - with blend motion tracking
        """
        # If we commanded a blended move, we are NOT idle until the robot's state
        # change from MOVING back to IDLE confirms it's finished.
        if self.is_blended_move_active:
            return False
            
        return self.cmd_confirm_flag and self.sys_status.robot_state == RobotState.IDLE.value
    
    def is_error(self) -> bool:
        """Check if there are any errors"""
        return (self.fatal_info_robot_connection_error or 
                self.fatal_info_robot_data_error or 
                self.debug_miss_command_working_check_count > 2)
    
    def clear_errors(self):
        """Clear all error flags"""
        self.fatal_info_robot_connection_error = False
        self.fatal_info_robot_miss_command_working_check = False
        self.fatal_info_robot_data_error = False
        self.super_fatal_error_robot_miss_command_working_check = False
        self.super_fatal_error_robot_data = False
        self.debug_miss_command_working_check_count = 0
        
        self.logger.info("Error flags cleared")
    
    # --- COMMAND SENDING METHOD ---
    def write_command(self, command: str, is_move_command: bool = False) -> bool:
        """Write command to robot using sendall for robustness."""
        if not self.cmd_connection_status or not self.cmd_socket:
            self.logger.warning("Cannot send command - not connected to command port")
            return False
        
        try:
            if is_move_command:
                self.move_cmd_flag = True
            
            self.cmd_confirm_flag = False
            # Use sendall to ensure the entire command is transmitted
            self.cmd_socket.sendall(command.encode('utf-8'))
            self.logger.info(f"Sent command: {command.strip()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send command: {e}")
            self.disconnect_cmd()
            return False
    
    # ============================================================================
    # Robot Control Commands
    # ============================================================================
    
    def robot_init(self) -> bool:
        """Initialize robot"""
        return self.write_command("mc jall init ")
    
    def set_program_mode_real(self) -> bool:
        """Set robot to real mode"""
        return self.write_command("pgmode real ")
    
    def set_program_mode_simulation(self) -> bool:
        """Set robot to simulation mode"""
        return self.write_command("pgmode simulation ")
    
    def move_joint(self, joints: List[float], speed: float = 1.0, acceleration: float = 1.0) -> bool:
        """Move robot joints"""
        if len(joints) != 6:
            self.logger.error("Joint move requires exactly 6 joint values")
            return False
        
        command = f"jointall {speed:.3f}, {acceleration:.3f}, {joints[0]:.3f}, {joints[1]:.3f}, {joints[2]:.3f}, {joints[3]:.3f}, {joints[4]:.3f}, {joints[5]:.3f}"
        
        if self.write_command(command):
            self.move_cmd_flag = False
            self.sys_status.robot_state = RobotState.MOVING.value
            return True
        return False
    
    def move_tcp(self, position: List[float], speed: float = 1.0, acceleration: float = 1.0) -> bool:
        """Move robot TCP"""
        if len(position) != 6:
            self.logger.error("TCP move requires exactly 6 position values")
            return False
        
        command = f"movetcp {speed:.3f}, {acceleration:.3f}, {position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}, {position[3]:.3f}, {position[4]:.3f}, {position[5]:.3f}"
        
        if self.write_command(command, True):
            self.sys_status.robot_state = RobotState.MOVING.value
            return True
        return False
    
    # ============================================================================
    # BLEND MOTION METHODS
    # ============================================================================
    
    def move_joint_blend_clear(self) -> bool:
        """
        Clear joint blend motion points
        """
        self.logger.info("Clearing joint blend motion points")
        return self.write_command("blend_jnt clear_pt ")
    
    def move_joint_blend_add_point(self, joints: List[float], speed: float = 1.0, acceleration: float = 1.0) -> bool:
        """
        Add joint blend motion point
        
        Args:
            joints: List of 6 joint angles in degrees
            speed: Motion speed (default 1.0)
            acceleration: Motion acceleration (default 1.0)
        """
        if len(joints) != 6:
            self.logger.error("Joint blend add point requires exactly 6 joint values")
            return False
        
        command = f"blend_jnt add_pt {speed:.3f}, {acceleration:.3f}, {joints[0]:.3f}, {joints[1]:.3f}, {joints[2]:.3f}, {joints[3]:.3f}, {joints[4]:.3f}, {joints[5]:.3f} "
        
        if self.write_command(command):
            self.sys_status.robot_state = RobotState.MOVING.value
            self.logger.info(f"Added joint blend point: {joints}")
            return True
        return False
    
    def move_joint_blend_move_point(self) -> bool:
        """
        Execute joint blend motion
        """
        self.logger.info("Executing joint blend motion")
        if self.write_command("blend_jnt move_pt ", True):
            self.move_cmd_flag = True
            self.is_blended_move_active = True
            self.sys_status.robot_state = RobotState.MOVING.value
            return True
        return False
    
    def move_tcp_blend_clear(self) -> bool:
        """
        Clear TCP blend motion points
        """
        self.logger.info("Clearing TCP blend motion points")
        return self.write_command("blend_tcp clear_pt ")
    
    def move_tcp_blend_add_point(self, radius: float, position: List[float], speed: float = 1.0, acceleration: float = 1.0) -> bool:
        """
        Add TCP blend motion point
        
        Args:
            radius: Blend radius
            position: List of 6 TCP position values [x, y, z, rx, ry, rz]
            speed: Motion speed (default 1.0)
            acceleration: Motion acceleration (default 1.0)
        """
        if len(position) != 6:
            self.logger.error("TCP blend add point requires exactly 6 position values")
            return False
        
        command = f"blend_tcp add_pt {speed:.3f}, {acceleration:.3f}, {radius:.3f}, {position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}, {position[3]:.3f}, {position[4]:.3f}, {position[5]:.3f} "
        
        if self.write_command(command):
            self.sys_status.robot_state = RobotState.MOVING.value
            self.logger.info(f"Added TCP blend point: radius={radius}, position={position}")
            return True
        return False
    
    def move_tcp_blend_move_point(self) -> bool:
        """
        Execute TCP blend motion
        """
        self.logger.info("Executing TCP blend motion")
        if self.write_command("blend_tcp move_pt ", True):
            self.move_cmd_flag = True
            self.is_blended_move_active = True
            self.sys_status.robot_state = RobotState.MOVING.value
            return True
        return False
    
    # ============================================================================
    # CIRCLE MOTION METHODS
    # ============================================================================
    
    def move_circle_three_point(self, motion_type: CircleMotionType, point1: List[float], point2: List[float], speed: float = 1.0, acceleration: float = 1.0) -> bool:
        """
        Move robot in circular motion using three points
        
        Args:
            motion_type: The type of motion (e.g., CircleMotionType.INTENDED)
            point1: First point [x1, y1, z1, rx1, ry1, rz1]
            point2: Second point [x2, y2, z2, rx2, ry2, rz2]
            speed: Motion speed (default 1.0)
            acceleration: Motion acceleration (default 1.0)
        """
        if len(point1) != 6 or len(point2) != 6:
            self.logger.error("Circle three point requires exactly 6 values for each point")
            return False
        
        # Use Enum to get string name, making it safer
        type_str = motion_type.name.lower()
        
        command = (f"movecircle absolute threepoints {type_str} {speed:.3f}, {acceleration:.3f}, "
                  f"{point1[0]:.3f}, {point1[1]:.3f}, {point1[2]:.3f}, {point1[3]:.3f}, {point1[4]:.3f}, {point1[5]:.3f}, "
                  f"{point2[0]:.3f}, {point2[1]:.3f}, {point2[2]:.3f}, {point2[3]:.3f}, {point2[4]:.3f}, {point2[5]:.3f} ")
        
        if self.write_command(command, True):
            self.move_cmd_flag = True
            self.sys_status.robot_state = RobotState.MOVING.value
            self.logger.info(f"Executing circle motion (three points): type={type_str}, point1={point1}, point2={point2}")
            return True
        return False
    
    def move_circle_axis(self, motion_type: CircleMotionType, center: List[float], axis: List[float], rotation_angle: float, speed: float = 1.0, acceleration: float = 1.0) -> bool:
        """
        Move robot in circular motion around axis
        
        Args:
            motion_type: The type of motion (e.g., CircleMotionType.CONSTANT)
            center: Center point [cx, cy, cz]
            axis: Axis vector [ax, ay, az]
            rotation_angle: Rotation angle in degrees
            speed: Motion speed (default 1.0)
            acceleration: Motion acceleration (default 1.0)
        """
        if len(center) != 3 or len(axis) != 3:
            self.logger.error("Circle axis requires exactly 3 values for center and axis")
            return False
        
        # Use Enum to get string name, making it safer
        type_str = motion_type.name.lower()
        
        command = (f"movecircle absolute axis {type_str} {speed:.3f}, {acceleration:.3f}, {rotation_angle:.3f}, "
                  f"{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}, "
                  f"{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f} ")
        
        if self.write_command(command, True):
            self.move_cmd_flag = True
            self.sys_status.robot_state = RobotState.MOVING.value
            self.logger.info(f"Executing circle motion (axis): type={type_str}, center={center}, axis={axis}, angle={rotation_angle}")
            return True
        return False
    
    def cobot_init(self) -> bool:
        """
        Initialize robot
        Required by authentic human imitation system
        """
        self.logger.info("Initializing robot (cobot_init)")
        return self.robot_init()
    
    def program_mode_real(self) -> bool:
        """
        Set robot to real mode
        Required by authentic human imitation system
        """
        self.logger.info("Setting robot to real mode (program_mode_real)")
        return self.set_program_mode_real()
    
    def motion_halt(self) -> bool:
        """
        Emergency stop robot motion
        """
        self.logger.warning("Emergency halt commanded (motion_halt)")
        success = self.write_command("task stop ")
        if success:
            self.sys_status.robot_state = RobotState.PAUSED_OR_STOPPED.value
            self.logger.info("Robot motion halted")
        else:
            self.logger.error("Failed to halt robot motion")
        return success
    
    def motion_pause(self) -> bool:
        """
        Pause robot motion
        """
        self.logger.info("Pausing robot motion")
        success = self.write_command("task pause ")
        if success:
            self.sys_status.robot_state = RobotState.PAUSED_OR_STOPPED.value
            self.logger.info("Robot motion paused")
        return success
    
    def motion_resume(self) -> bool:
        """Resume robot motion"""
        self.logger.info("Resuming robot motion")
        success = self.write_command("task resume_a ")
        if success:
            self.logger.info("Robot motion resumed")
        return success
    
    def motion_play(self) -> bool:
        """
        Start/play robot motion
        """
        self.logger.info("Starting robot motion")
        return self.write_command("task play ")
    
    def collision_resume(self) -> bool:
        """Resume after collision"""
        self.logger.info("Resuming after collision")
        return self.write_command("task resume_b ")
    
    def set_system_forced_stop_flag(self):
        """
        Set forced stop flag
        """
        self.system_forced_stop_flag = True
        self.logger.warning("System forced stop flag SET")
    
    def clear_system_forced_stop_flag(self):
        """
        Clear forced stop flag
        """
        self.system_forced_stop_flag = False
        self.logger.info("System forced stop flag CLEARED")
    
    def control_box_dout(self, port: int, value: int) -> bool:
        """Control digital output"""
        command = f"set_box_dout({port}, {value}) "
        return self.write_command(command)
    
    def control_box_digital_out(self, *values) -> bool:
        """Control all digital outputs"""
        if len(values) != 16:
            self.logger.error("ControlBoxDigitalOut requires exactly 16 values")
            return False
        
        value_str = ", ".join(str(v) for v in values)
        command = f"digital_out {value_str}"
        return self.write_command(command)
    
    def control_box_analog_out(self, a0: float, a1: float, a2: float, a3: float) -> bool:
        """Control analog outputs"""
        command = f"analog_out {a0:.3f}, {a1:.3f}, {a2:.3f}, {a3:.3f} "
        return self.write_command(command)
    
    def tool_out(self, voltage: int, d0: int, d1: int) -> bool:
        """Control tool outputs"""
        command = f"tool_out {voltage}, {d0}, {d1} "
        return self.write_command(command)
    
    def base_speed_change(self, speed: float) -> bool:
        """Change base speed"""
        command = f"sdw default_speed {speed:.3f} "
        return self.write_command(command)
    
    # ============================================================================
    # SAFETY FEATURES
    # ============================================================================
    
    def _update_forced_stop_logic(self):
        """
        Forced stop digital output logic
        """
        try:
            if not self.system_forced_stop_flag:
                # Normal operation - ensure digital output 0 is OFF
                if len(self.sys_status.digital_out) > 0 and self.sys_status.digital_out[0] == 1:
                    self.control_box_dout(0, 0)
            else:
                # Forced stop active - ensure digital output 0 is ON
                if len(self.sys_status.digital_out) > 0 and self.sys_status.digital_out[0] == 0:
                    self.control_box_dout(0, 1)
        except Exception as e:
            self.logger.error(f"Error in forced stop logic: {e}")
    
    # ============================================================================
    # Update Loop and Data Handling
    # ============================================================================
    
    def _update_loop(self):
        """Main update loop with enhanced safety features"""
        while self.running:
            try:
                # Handle connections
                if not self.cmd_connection_status:
                    self.connect_cmd()
                if not self.data_connection_status:
                    self.connect_data()
                
                # Assume a connection error until proven otherwise in this loop
                not_connected = 0
                if not self.cmd_connection_status:
                    not_connected += 1
                if not self.data_connection_status:
                    not_connected += 1
                
                if not_connected > 0:
                    if not self.fatal_info_robot_connection_error:
                         self.logger.warning("Robot connection error: One or more sockets are disconnected.")
                    self.fatal_info_robot_connection_error = True
                else:
                    self.fatal_info_robot_connection_error = False

                # Handle data reception and command responses if connected
                if self.data_connection_status:
                    # ### MODIFIED SECTION START ###
                    # Increment data counter and check for timeout
                    self.data_recv_count += 1
                    if self.data_recv_count > 20: # Timeout is 20 * 100ms = 2s
                        if not self.fatal_info_robot_data_error:
                            self.logger.error("FATAL: No data received from robot for over 2 seconds.")
                        self.fatal_info_robot_data_error = True
                    else:
                        self.fatal_info_robot_data_error = False # Reset if count is not over threshold
                    # ### MODIFIED SECTION END ###
                    
                    self._handle_data_reception()
                    # Add forced stop logic
                    self._update_forced_stop_logic()
                
                if self.cmd_connection_status:
                    self._handle_command_response()
                
                # Handle motion server
                self._handle_motion_server()
                
                # Call status update callback
                if self.status_update_callback and self.data_connection_status:
                    self.status_update_callback(self.sys_status)
                
                # Sleep for update interval
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Update loop error: {e}")
                if self.error_callback:
                    self.error_callback("update_loop", str(e))
                time.sleep(1.0)  # Longer sleep on error
    
    def _handle_data_reception(self):
        """Handle data reception from robot"""
        if not self.data_socket:
            return
        
        try:
            # Request data from robot
            self.data_socket.sendall(b'reqdata')

            self.data_socket.settimeout(0.1)
            data = self.data_socket.recv(4096)
            if data:
                self.recv_buffer.extend(data)
                self._parse_received_data()
        except socket.timeout:
            pass
        except Exception as e:
            self.logger.error(f"Data reception error: {e}")
            self.disconnect_data()
    
    def _parse_received_data(self):
        """Parse received data packets"""
        while len(self.recv_buffer) > 4:
            if self.recv_buffer[0] == ord('$'):
                size = (self.recv_buffer[2] << 8) | self.recv_buffer[1]
                
                if size <= len(self.recv_buffer):
                    packet_type = self.recv_buffer[3]
                    
                    if packet_type == 3:  # System status
                        self.data_recv_count = 0 # Reset counter on successful data packet
                        if self.move_cmd_cnt > 0:
                            self.move_cmd_cnt -= 1
                        else:
                            try:
                                from robot_data import RobotDataParser
                                self.sys_status = RobotDataParser.parse_system_status(self.recv_buffer[:size])
                                
                                # Blend motion completion detection
                                current_state = self.sys_status.robot_state
                                if self.is_blended_move_active:
                                    if (self.previous_robot_state == RobotState.MOVING.value and 
                                        current_state == RobotState.IDLE.value and 
                                        self.cmd_confirm_flag):
                                        self.logger.info("Blended move completed")
                                        self.is_blended_move_active = False
                                
                                self.previous_robot_state = current_state

                                # AI1 connection check
                                if len(self.sys_status.analog_in) > 1:
                                    self.ai1_connected = 1 if self.sys_status.analog_in[1] < 1.0 else 0
                                
                                # Command out flag check
                                if self.command_out_flag == 1:
                                    if self.sys_status.robot_state == 3:  # Moving state
                                        self.command_out_flag = 0
                                        
                            except Exception as e:
                                self.logger.error(f"Failed to parse system status: {e}")
                        
                        self.recv_buffer = self.recv_buffer[size:]
                    
                    elif packet_type == 4:  # System config
                        try:
                            from robot_data import RobotDataParser
                            self.sys_config = RobotDataParser.parse_system_config(self.recv_buffer[:size])
                        except Exception as e:
                            self.logger.error(f"Failed to parse system config: {e}")
                        self.recv_buffer = self.recv_buffer[size:]
                    
                    elif packet_type == 10:  # System popup
                        try:
                            from robot_data import RobotDataParser
                            self.sys_popup = RobotDataParser.parse_system_popup(self.recv_buffer[:size])
                        except Exception as e:
                            self.logger.error(f"Failed to parse system popup: {e}")
                        self.recv_buffer = self.recv_buffer[size:]
                    
                    else:
                        self.recv_buffer = self.recv_buffer[1:]
                else:
                    break
            else:
                self.recv_buffer = self.recv_buffer[1:]
    
    def _handle_command_response(self):
        """Handle command response from robot"""
        if not self.cmd_socket:
            return
        
        try:
            self.cmd_socket.settimeout(0.1)
            data = self.cmd_socket.recv(1024)
            if data:
                response = data.decode('utf-8', errors='ignore')
                if "The command was executed\n" in response:
                    self.cmd_confirm_flag = True
                    if self.move_cmd_flag:
                        self.move_cmd_cnt = 5
                        self.sys_status.robot_state = RobotState.MOVING.value
                        self.move_cmd_flag = False
        except socket.timeout:
            pass
        except Exception as e:
            self.logger.error(f"Command response error: {e}")
            self.disconnect_cmd()
    
    def _handle_motion_server(self):
        """Handle motion server communication"""
        if self.motion_server.has_data():
            data_packets = self.motion_server.get_received_data()
            for data in data_packets:
                try:
                    message = data.decode('utf-8', errors='ignore').strip()
                    if message == "ALIVE":
                        pass  # Handle keep-alive
                    elif message == "MOTION_DONE":
                        self.logger.info("Motion completed")
                        self.robot_moving = False
                except Exception as e:
                    self.logger.error(f"Motion server data error: {e}")
    
    # Motion server callbacks
    def _on_motion_server_connected(self):
        """Motion server connection callback"""
        self.logger.info("Motion server client connected")
    
    def _on_motion_server_disconnected(self):
        """Motion server disconnection callback"""
        self.logger.info("Motion server client disconnected")
    
    def _on_motion_server_data(self, data: bytes):
        """Motion server data received callback"""
        pass  # Data is automatically handled in _handle_motion_server

    # Getters for robot status
    def get_robot_cmd_connection(self) -> bool:
        """Get command connection status"""
        return self.cmd_connection_status
    
    def get_robot_data_connection(self) -> bool:
        """Get data connection status"""
        return self.data_connection_status
    
    def get_robot_ai1_connected(self) -> bool:
        """Get AI1 connection status"""
        return self.ai1_connected

# Example usage
# if __name__ == "__main__":
#     # Create robot driver
#     robot = RobotArmDriver()
    
#     # Set up callbacks
#     def on_status_update(status):
#         print(f"Robot state: {status.robot_state}, Joint 0: {status.jnt_ang[0]:.2f}")
    
#     def on_connection_change(connection_type, connected):
#         print(f"Connection {connection_type}: {'Connected' if connected else 'Disconnected'}")
    
#     def on_error(error_type, message):
#         print(f"Error [{error_type}]: {message}")
    
#     robot.set_status_update_callback(on_status_update)
#     robot.set_connection_status_callback(on_connection_change)
#     robot.set_error_callback(on_error)
    
#     # Start robot driver
#     if robot.start():
#         print("Robot driver started successfully")
        
#         try:
#             # Example commands
#             time.sleep(2)  # Wait for connections
            
#             # Initialize robot
#             robot.cobot_init()  # Using new compatibility method
#             time.sleep(1)

#             # Set to real mode
#             robot.program_mode_real()  # Using new compatibility method
#             time.sleep(1)
            
#             # Move joints
#             robot.move_joint([0, 0, 0, 0, 0, 0], speed=0.5)
#             time.sleep(3)
            
#             # Move TCP
#             robot.move_tcp([100, 200, 300, 0, 0, 0], speed=0.3)
#             time.sleep(3)
            
#             # NEW: Demonstrate blend motion capabilities
#             print("Testing blend motion...")
#             robot.move_joint_blend_clear()
#             robot.move_joint_blend_add_point([0, 0, 0, 0, 0, 0], speed=0.5)
#             robot.move_joint_blend_add_point([10, 20, 30, 0, 0, 0], speed=0.5)
#             robot.move_joint_blend_add_point([20, 40, 60, 0, 0, 0], speed=0.5)
#             robot.move_joint_blend_move_point()
            
#             # Wait for blend motion to complete
#             while not robot.is_motion_idle():
#                 time.sleep(0.1)
#             print("Blend motion completed")
            
#             # NEW: Demonstrate circle motion capabilities (using Enum)
#             print("Testing circle motion...")
#             point1 = [100, 100, 200, 0, 0, 0]
#             point2 = [200, 200, 200, 0, 0, 0]
#             robot.move_circle_three_point(CircleMotionType.INTENDED, point1, point2, speed=0.3)
            
#             # Wait for circle motion to complete
#             while not robot.is_motion_idle():
#                 time.sleep(0.1)
#             print("Circle motion completed")
            
#             # Keep running
#             input("Press Enter to stop...")
            
#         except KeyboardInterrupt:
#             print("Interrupted by user")
        
#         finally:
#             robot.stop()
#     else:
#         print("Failed to start robot driver")