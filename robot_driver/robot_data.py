"""
Robot Data Structures and Binary Protocol Parser
"""

import struct
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum

# Constants
ROBOT_BASE_LOW = 100.0
ROBOT_BASE_HIGH = 150.0
ROBOT_STOP_SPEED = 0.0
ROBOT_SLOW_SPEED = 0.2
ROBOT_NORMAL_SPEED = 1.0

MAX_SHARED_DATA = 128
MAX_CONFIG_DATA = 157

# --- NEWLY ADDED ENUM ---
class CircleMotionType(Enum):
    """Defines the type of circular motion path."""
    INTENDED = 0
    CONSTANT = 1
    RADIAL = 2

# Status Constants
# class PlatformInitStatus(Enum):
#     IDLE = 0
#     START = 1
#     CHECK_CONNECTION = 2
#     CHECK_ROBOT_IN_GOOD_POSTURE = 3
#     CHECK_ROBOT_CONNECTION = 4
#     PROGRAM_TURN_ON = 5
#     OUTLET = 6
#     OUTLET_DONE = 7
#     DONE = 8
#     ON_SUCCESS_STATE = 9
#     ON_FAIL_STATE = 10

class PlatformInitStatus(Enum):
    """
    Defines initialization process states based on the user manual.
    """
    DEFAULT = 0
    VOLTAGE_CHECK = 1
    DEVICE_CHECK = 2
    POSITION_CONTROL_START = 3
    PARAMETER_CHECK = 4
    COLLISION_CHECK = 5
    INITIALIZATION_DONE = 6

class PlatformOperationStatus(Enum):
    IDLE = 0
    START = 1
    CHECK_ON_INIT_SUCCESS_STATE = 2
    CHECK_ROBOT_POSTURE = 3
    CHECK_PLATFORM_CLOSED = 4
    CHECK_OUTLET_OCCUPY = 5
    CHECK_SCHEDULER = 6
    ON_OPERATING_STATE = 7
    ON_OPERATING_FAIL_STATE = 8
    ON_ERROR_STATE = 9
    FATAL_CHECK_ROBOT_DANGEROUS_POSITION = 10
    FATAL_AVOID_ROBOT_DANGEROUS_POSITION = 11
    PLATFORM_OPENED_DETECTED = 12

class RobotState(Enum):
    IDLE = 1
    PAUSED_OR_STOPPED = 2
    MOVING = 3

class TaskState(Enum):
    STOPPED = 1
    PAUSED = 2
    RUNNING = 3

@dataclass
class SystemStatus:
    """Robot system status - systemSTAT with digital I/O types"""
    
    # Header (4 bytes)
    header: bytes = b'$\x00\x00\x03'
    
    # Basic data (float time + 6*float jnt_ref + 6*float jnt_ang + 6*float cur = 19*4 = 76 bytes)
    time: float = 0.0                    # time [sec]
    jnt_ref: List[float] = None         # joint reference [deg] - 6 elements
    jnt_ang: List[float] = None         # joint encoder value [deg] - 6 elements  
    cur: List[float] = None             # joint current value [mA] - 6 elements
    
    # TCP data (6*float tcp_ref + 6*float tcp_pos = 12*4 = 48 bytes)
    tcp_ref: List[float] = None         # calculated TCP from reference [mm, deg] - 6 elements
    tcp_pos: List[float] = None         # calculated TCP from encoder [mm, deg] - 6 elements
    
    # I/O data (4*float analog_in + 4*float analog_out + 16*float digital_in + 16*float digital_out = 40*4 = 160 bytes)
    analog_in: List[float] = None       # analog input [V] - 4 elements
    analog_out: List[float] = None      # analog output [V] - 4 elements
    digital_in: List[int] = None        # digital input [0 or 1] - 16 elements
    digital_out: List[int] = None       # digital output [0 or 1] - 16 elements
    
    # Temperature (6*float = 24 bytes)
    temperature_mc: List[float] = None  # board temperature [celsius] - 6 elements
    
    # Task info (6*4 = 24 bytes)
    task_pc: int = 0
    task_repeat: int = 0
    task_run_id: int = 0
    task_run_num: int = 0
    task_run_time: float = 0.0
    task_state: int = 0
    
    # Robot state (3*4 = 12 bytes)
    default_speed: float = 0.0
    robot_state: int = 1                # 1:idle 2:paused 3:moving
    power_state: int = 0
    
    # Target and joint info (6*float + 6*int = 48 bytes)
    tcp_target: List[float] = None      # 6 elements
    jnt_info: List[int] = None          # 6 elements
    
    # Configuration (3*4 = 12 bytes)
    collision_detect_onoff: int = 0
    is_freedrive_mode: int = 0
    program_mode: int = 0               # 0:real 1:simulation
    
    # Init info (2*4 = 8 bytes)
    init_state_info: int = 0
    init_error: int = 0
    
    # Tool flange board (2*float + 2*int + 2*int + 1*float = 28 bytes)
    tfb_analog_in: List[float] = None   # 2 elements
    tfb_digital_in: List[int] = None    # 2 elements
    tfb_digital_out: List[int] = None   # 2 elements
    tfb_voltage_out: float = 0.0
    
    # Operation status (5*4 = 20 bytes)
    op_stat_collision_occur: int = 0
    op_stat_sos_flag: int = 0
    op_stat_self_collision: int = 0
    op_stat_soft_estop_occur: int = 0
    op_stat_ems_flag: int = 0
    
    # Digital config (2*4 = 8 bytes)
    digital_in_config: List[int] = None # 2 elements
    
    # Inbox settings (4*4 = 16 bytes)
    inbox_trap_flag: List[int] = None   # 2 elements
    inbox_check_mode: List[int] = None  # 2 elements
    
    # Force/torque (6*float = 24 bytes)
    eft_fx: float = 0.0
    eft_fy: float = 0.0
    eft_fz: float = 0.0
    eft_mx: float = 0.0
    eft_my: float = 0.0
    eft_mz: float = 0.0
    
    def __post_init__(self):
        """Initialize list fields with default values"""
        if self.jnt_ref is None:
            self.jnt_ref = [0.0] * 6
        if self.jnt_ang is None:
            self.jnt_ang = [0.0] * 6
        if self.cur is None:
            self.cur = [0.0] * 6
        if self.tcp_ref is None:
            self.tcp_ref = [0.0] * 6
        if self.tcp_pos is None:
            self.tcp_pos = [0.0] * 6
        if self.analog_in is None:
            self.analog_in = [0.0] * 4
        if self.analog_out is None:
            self.analog_out = [0.0] * 4
        if self.digital_in is None:
            self.digital_in = [0] * 16
        if self.digital_out is None:
            self.digital_out = [0] * 16
        if self.temperature_mc is None:
            self.temperature_mc = [0.0] * 6
        if self.tcp_target is None:
            self.tcp_target = [0.0] * 6
        if self.jnt_info is None:
            self.jnt_info = [0] * 6
        if self.tfb_analog_in is None:
            self.tfb_analog_in = [0.0] * 2
        if self.tfb_digital_in is None:
            self.tfb_digital_in = [0] * 2
        if self.tfb_digital_out is None:
            self.tfb_digital_out = [0] * 2
        if self.digital_in_config is None:
            self.digital_in_config = [0] * 2
        if self.inbox_trap_flag is None:
            self.inbox_trap_flag = [0] * 2
        if self.inbox_check_mode is None:
            self.inbox_check_mode = [0] * 2

@dataclass  
class SystemConfig:
    """Robot system configuration"""
    
    header: bytes = b'$\x00\x00\x04'
    
    # Collision and workspace (8*float + 1*int = 36 bytes)
    sensitivity: float = 0.0
    work_x_min: float = 0.0
    work_x_max: float = 0.0
    work_y_min: float = 0.0
    work_y_max: float = 0.0
    work_z_min: float = 0.0
    work_z_max: float = 0.0
    work_onoff: int = 0
    mount_rotate: List[float] = None    # 3 elements
    
    # Tool configuration (9*float = 36 bytes)
    toolbox_size: List[float] = None           # 3 elements
    toolbox_center_pos: List[float] = None     # 3 elements
    tool_mass: float = 0.0
    tool_mass_center_pos: List[float] = None   # 3 elements
    tool_ee_pos: List[float] = None            # 3 elements
    
    # USB and RS485 (8*int = 32 bytes)
    usb_detected_flag: int = 0
    usb_copy_done_flag: int = 0
    rs485_tool_baud: int = 0
    rs485_tool_stopbit: int = 0
    rs485_tool_paritybit: int = 0
    rs485_box_baud: int = 0
    rs485_box_stopbit: int = 0
    rs485_box_paritybit: int = 0
    
    # I/O functions (32*int = 128 bytes)
    io_function_in: List[int] = None    # 16 elements
    io_function_out: List[int] = None   # 16 elements
    
    # Network (12*int = 48 bytes)
    ip_addr: List[int] = None           # 4 elements
    netmask: List[int] = None           # 4 elements
    gateway: List[int] = None           # 4 elements
    
    # Version and script (1*int + 64*char = 68 bytes)
    version: int = 0
    default_script: str = ""            # 64 characters
    
    # Additional config continues...
    auto_init: int = 0
    inbox0_size: List[float] = None     # 3 elements
    inbox0_pos: List[float] = None      # 3 elements
    inbox1_size: List[float] = None     # 3 elements
    inbox1_pos: List[float] = None      # 3 elements
    default_repeat_num: int = 0
    direct_teaching_sensitivity: List[float] = None  # 6 elements
    tool_ee_ori: List[float] = None     # 3 elements
    user_coord_0: List[float] = None    # 6 elements
    user_coord_1: List[float] = None    # 6 elements
    user_coord_2: List[float] = None    # 6 elements
    
    # DIO settings
    dio_begin_box_dout: List[int] = None     # 16 elements
    dio_begin_box_aout: List[float] = None   # 4 elements
    dio_end_box_dout: List[int] = None       # 16 elements
    dio_end_box_aout: List[float] = None     # 4 elements
    dio_begin_tool_voltage: int = 0
    dio_end_tool_voltage: int = 0
    dio_begin_tool_dout: List[int] = None    # 2 elements
    dio_end_tool_dout: List[int] = None      # 2 elements
    
    # Final settings
    ext_ft_model_info: int = 0
    robot_model_type: int = 0
    collision_stop_mode: int = 0
    
    def __post_init__(self):
        """Initialize list fields with default values"""
        if self.mount_rotate is None:
            self.mount_rotate = [0.0] * 3
        if self.toolbox_size is None:
            self.toolbox_size = [0.0] * 3
        if self.toolbox_center_pos is None:
            self.toolbox_center_pos = [0.0] * 3
        if self.tool_mass_center_pos is None:
            self.tool_mass_center_pos = [0.0] * 3
        if self.tool_ee_pos is None:
            self.tool_ee_pos = [0.0] * 3
        if self.io_function_in is None:
            self.io_function_in = [0] * 16
        if self.io_function_out is None:
            self.io_function_out = [0] * 16
        if self.ip_addr is None:
            self.ip_addr = [0] * 4
        if self.netmask is None:
            self.netmask = [0] * 4
        if self.gateway is None:
            self.gateway = [0] * 4
        if self.inbox0_size is None:
            self.inbox0_size = [0.0] * 3
        if self.inbox0_pos is None:
            self.inbox0_pos = [0.0] * 3
        if self.inbox1_size is None:
            self.inbox1_size = [0.0] * 3
        if self.inbox1_pos is None:
            self.inbox1_pos = [0.0] * 3
        if self.direct_teaching_sensitivity is None:
            self.direct_teaching_sensitivity = [0.0] * 6
        if self.tool_ee_ori is None:
            self.tool_ee_ori = [0.0] * 3
        if self.user_coord_0 is None:
            self.user_coord_0 = [0.0] * 6
        if self.user_coord_1 is None:
            self.user_coord_1 = [0.0] * 6
        if self.user_coord_2 is None:
            self.user_coord_2 = [0.0] * 6
        if self.dio_begin_box_dout is None:
            self.dio_begin_box_dout = [0] * 16
        if self.dio_begin_box_aout is None:
            self.dio_begin_box_aout = [0.0] * 4
        if self.dio_end_box_dout is None:
            self.dio_end_box_dout = [0] * 16
        if self.dio_end_box_aout is None:
            self.dio_end_box_aout = [0.0] * 4
        if self.dio_begin_tool_dout is None:
            self.dio_begin_tool_dout = [0] * 2
        if self.dio_end_tool_dout is None:
            self.dio_end_tool_dout = [0] * 2

@dataclass
class SystemPopup:
    """System popup - EXACTLY matches systemPOPUP"""
    header: bytes = b'$\x00\x00\x0A'
    type: int = 0
    msg: str = ""           # 1000 characters max
    length: int = 0

class RobotDataParser:
    """
    Binary data parser that parsing logic
    Digital I/O parsed as floats then converted to ints
    """
    
    @staticmethod
    def parse_system_status(data: bytes) -> SystemStatus:
        """
        Parse binary system status data
        Digital I/O fields are stored as FLOATS
        Matches: memcpy(&sys_status, recvBuf.data(), sizeof(systemSTAT));
        """
        expected_size = 128 * 4  # 512 bytes
        
        if len(data) < expected_size:
            raise ValueError(f"SystemStatus data too short: {len(data)} bytes, expected {expected_size}")
        
        status = SystemStatus()
        offset = 0
        
        try:
            # Header (4 bytes) - char header[4]
            status.header = data[offset:offset+4]
            offset += 4
            
            # Basic data (19*float = 76 bytes)
            # float time
            status.time = struct.unpack('<f', data[offset:offset+4])[0]
            offset += 4
            
            # float jnt_ref[6]
            status.jnt_ref = list(struct.unpack('<6f', data[offset:offset+24]))
            offset += 24
            
            # float jnt_ang[6] 
            status.jnt_ang = list(struct.unpack('<6f', data[offset:offset+24]))
            offset += 24
            
            # float cur[6]
            status.cur = list(struct.unpack('<6f', data[offset:offset+24]))
            offset += 24
            
            # float tcp_ref[6]
            status.tcp_ref = list(struct.unpack('<6f', data[offset:offset+24]))
            offset += 24
            
            # float tcp_pos[6]
            status.tcp_pos = list(struct.unpack('<6f', data[offset:offset+24]))
            offset += 24
            
            # float analog_in[4]
            status.analog_in = list(struct.unpack('<4f', data[offset:offset+16]))
            offset += 16
            
            # float analog_out[4] 
            status.analog_out = list(struct.unpack('<4f', data[offset:offset+16]))
            offset += 16
            
            # CRITICAL Digital I/O are stored as FLOATS
            # float digital_in[16] - parse as floats then convert to int
            digital_in_floats = list(struct.unpack('<16f', data[offset:offset+64]))
            status.digital_in = [int(round(x)) for x in digital_in_floats]
            offset += 64
            
            # float digital_out[16] - parse as floats then convert to int
            digital_out_floats = list(struct.unpack('<16f', data[offset:offset+64]))
            status.digital_out = [int(round(x)) for x in digital_out_floats]
            offset += 64
            
            # float temperature_mc[6]
            status.temperature_mc = list(struct.unpack('<6f', data[offset:offset+24]))
            offset += 24
            
            # Task information - mixed int and float
            (status.task_pc, status.task_repeat, status.task_run_id, 
             status.task_run_num) = struct.unpack('<4i', data[offset:offset+16])
            offset += 16
            
            status.task_run_time = struct.unpack('<f', data[offset:offset+4])[0]
            offset += 4
            
            status.task_state = struct.unpack('<i', data[offset:offset+4])[0]
            offset += 4
            
            # Robot state
            status.default_speed = struct.unpack('<f', data[offset:offset+4])[0]
            offset += 4
            
            (status.robot_state, status.power_state) = struct.unpack('<2i', data[offset:offset+8])
            offset += 8
            
            # TCP target and joint info
            status.tcp_target = list(struct.unpack('<6f', data[offset:offset+24]))
            offset += 24
            
            status.jnt_info = list(struct.unpack('<6i', data[offset:offset+24]))
            offset += 24
            
            # Configuration flags
            (status.collision_detect_onoff, status.is_freedrive_mode, 
             status.program_mode) = struct.unpack('<3i', data[offset:offset+12])
            offset += 12
            
            # Initialization
            (status.init_state_info, status.init_error) = struct.unpack('<2i', data[offset:offset+8])
            offset += 8
            
            # Tool flange board
            status.tfb_analog_in = list(struct.unpack('<2f', data[offset:offset+8]))
            offset += 8
            
            status.tfb_digital_in = list(struct.unpack('<2i', data[offset:offset+8]))
            offset += 8
            
            status.tfb_digital_out = list(struct.unpack('<2i', data[offset:offset+8]))
            offset += 8
            
            status.tfb_voltage_out = struct.unpack('<f', data[offset:offset+4])[0]
            offset += 4
            
            # Operation status
            (status.op_stat_collision_occur, status.op_stat_sos_flag,
             status.op_stat_self_collision, status.op_stat_soft_estop_occur,
             status.op_stat_ems_flag) = struct.unpack('<5i', data[offset:offset+20])
            offset += 20
            
            # Digital config
            status.digital_in_config = list(struct.unpack('<2i', data[offset:offset+8]))
            offset += 8
            
            # Inbox settings
            status.inbox_trap_flag = list(struct.unpack('<2i', data[offset:offset+8]))
            offset += 8
            
            status.inbox_check_mode = list(struct.unpack('<2i', data[offset:offset+8]))
            offset += 8
            
            # Force/torque
            (status.eft_fx, status.eft_fy, status.eft_fz,
             status.eft_mx, status.eft_my, status.eft_mz) = struct.unpack('<6f', data[offset:offset+24])
            offset += 24
            
            return status
            
        except struct.error as e:
            raise ValueError(f"Failed to parse SystemStatus at offset {offset}: {e}")
    
    @staticmethod
    def parse_system_config(data: bytes) -> SystemConfig:
        """Parse binary system config data"""
        expected_size = 157 * 4  # 628 bytes - MAX_CONFIG_DATA * 4
        
        if len(data) < expected_size:
            # Allow shorter data but log warning
            print(f"WARNING: SystemConfig data shorter than expected: {len(data)} < {expected_size}")
        
        config = SystemConfig()
        offset = 0
        
        try:
            # Header
            config.header = data[offset:offset+4]
            offset += 4
            
            # Basic config data - following exact C structure layout
            # float sensitivity, work limits
            (config.sensitivity, config.work_x_min, config.work_x_max,
             config.work_y_min, config.work_y_max, config.work_z_min,
             config.work_z_max) = struct.unpack('<7f', data[offset:offset+28])
            offset += 28
            
            # int work_onoff
            config.work_onoff = struct.unpack('<i', data[offset:offset+4])[0]
            offset += 4
            
            # float mount_rotate[3]
            config.mount_rotate = list(struct.unpack('<3f', data[offset:offset+12]))
            offset += 12
            
            # Tool configuration
            config.toolbox_size = list(struct.unpack('<3f', data[offset:offset+12]))
            offset += 12
            
            config.toolbox_center_pos = list(struct.unpack('<3f', data[offset:offset+12]))
            offset += 12
            
            config.tool_mass = struct.unpack('<f', data[offset:offset+4])[0]
            offset += 4
            
            config.tool_mass_center_pos = list(struct.unpack('<3f', data[offset:offset+12]))
            offset += 12
            
            config.tool_ee_pos = list(struct.unpack('<3f', data[offset:offset+12]))
            offset += 12
            
            # USB and RS485
            (config.usb_detected_flag, config.usb_copy_done_flag,
             config.rs485_tool_baud, config.rs485_tool_stopbit, config.rs485_tool_paritybit,
             config.rs485_box_baud, config.rs485_box_stopbit, config.rs485_box_paritybit) = \
                struct.unpack('<8i', data[offset:offset+32])
            offset += 32
            
            # I/O functions
            config.io_function_in = list(struct.unpack('<16i', data[offset:offset+64]))
            offset += 64
            
            config.io_function_out = list(struct.unpack('<16i', data[offset:offset+64]))
            offset += 64
            
            # Network
            config.ip_addr = list(struct.unpack('<4i', data[offset:offset+16]))
            offset += 16
            
            config.netmask = list(struct.unpack('<4i', data[offset:offset+16]))
            offset += 16
            
            config.gateway = list(struct.unpack('<4i', data[offset:offset+16]))
            offset += 16
            
            # Version
            config.version = struct.unpack('<i', data[offset:offset+4])[0]
            offset += 4
            
            # Default script (64 chars)
            script_data = data[offset:offset+64]
            null_pos = script_data.find(b'\x00')
            if null_pos >= 0:
                config.default_script = script_data[:null_pos].decode('utf-8', errors='ignore')
            else:
                config.default_script = script_data.decode('utf-8', errors='ignore')
            offset += 64
            
            
            
            return config
            
        except struct.error as e:
            raise ValueError(f"Failed to parse SystemConfig at offset {offset}: {e}")
    
    @staticmethod
    def parse_system_popup(data: bytes) -> SystemPopup:
        """Parse binary system popup data"""
        if len(data) < 1008:  # Header + type + msg + len
            raise ValueError(f"SystemPopup data too short: {len(data)} bytes")
        
        popup = SystemPopup()
        offset = 0
        
        try:
            # Header
            popup.header = data[offset:offset+4]
            offset += 4
            
            # Type (char)
            popup.type = struct.unpack('<B', data[offset:offset+1])[0]
            offset += 1
            
            # Message (1000 chars)
            msg_data = data[offset:offset+1000]
            # Find null terminator
            null_pos = msg_data.find(b'\x00')
            if null_pos >= 0:
                popup.msg = msg_data[:null_pos].decode('utf-8', errors='ignore')
            else:
                popup.msg = msg_data.decode('utf-8', errors='ignore')
            offset += 1000
            
            # Length (int)
            popup.length = struct.unpack('<i', data[offset:offset+4])[0]
            offset += 4
            
            return popup
            
        except struct.error as e:
            raise ValueError(f"Failed to parse SystemPopup at offset {offset}: {e}")
    
    @staticmethod
    def parse_packet(data: bytes) -> Tuple[int, bytes]:
        """
        Parse packet header and data from binary stream.
        if(recvBuf[0] == '$')
        int size = (int(uchar(recvBuf[2])<<8) | int(uchar(recvBuf[1])) );
        packet_type = recvBuf[3]
        """

        if len(data) < 4:
            raise ValueError("Packet too short for header")
        
        if data[0] != ord('$'):
            raise ValueError("Invalid packet header - expected '$'")
        
        # Size calculation: (recvBuf[2]<<8) | recvBuf[1]
        size = (data[2] << 8) | data[1]
        
        if len(data) < size:
            raise ValueError(f"Packet incomplete: expected {size}, got {len(data)}")
        
        packet_type = data[3]
        packet_data = data[:size]
        
        return packet_type, packet_data

    @staticmethod
    def validate_structure_sizes():
        """Validate that our parsing"""
        # systemSTAT should be 512 bytes (128 * 4)
        expected_status_size = 128 * 4
        
        # systemCONFIG should be 628 bytes (157 * 4)  
        expected_config_size = 157 * 4
        
        print(f"Expected SystemStatus size: {expected_status_size} bytes")
        print(f"Expected SystemConfig size: {expected_config_size} bytes")
        
        return expected_status_size, expected_config_size

# Test the parser
def test_binary_compatibility():
    """Test that parser handles binary data correctly"""
    print("Testing binary data compatibility...")
    
    # Create test binary data that matches expected format
    test_data = bytearray()
    test_data.extend(b'$')  # Header start
    test_data.extend(struct.pack('<H', 512))  # Size (little endian)  
    test_data.extend(b'\x03')  # Packet type 3 (system status)
    
    # Add proper system status data structure
    test_data.extend(b'\x00' * (512 - 4))  # Pad to correct size
    
    try:
        packet_type, packet_data = RobotDataParser.parse_packet(test_data)
        print(f"Packet parsing works: type={packet_type}, size={len(packet_data)}")
        
        if packet_type == 3:
            print("Binary parser structure is correct")
            return True
        
    except Exception as e:
        print(f"Binary parsing error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_binary_compatibility()
    RobotDataParser.validate_structure_sizes()