#!/usr/bin/env python3
"""
Clean Robot Motion Planning Interface

This module provides a clean, consolidated interface for robot motion planning with:
- Real-world units (mm, degrees) for input/output
- Direct robot API compatibility
- Automatic unit conversions
- JSON robot data compatibility

Compatible with robot API methods:
- robot.move_joint(q1, q2, q3, q4, q5, q6)  # degrees
- robot.move_tcp(x, y, z, rx, ry, rz)       # mm, degrees
- robot.blend_point([joints], radius)       # degrees, mm
"""

import sys
import os
import numpy as np
import json
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'kinematics', 'src'))

try:
    from forward_kinematic import ForwardKinematics
    from inverse_kinematic import InverseKinematics
    from motion_planner import MotionPlanner, PlanningStatus, PlanningStrategy
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

@dataclass
class RobotPose:
    """Robot pose in real-world units (mm, degrees)."""
    position_mm: List[float]     # [x, y, z] in millimeters
    orientation_deg: List[float] # [rx, ry, rz] in degrees (RPY)
    
    def to_transformation_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix (internal units: meters, radians)."""
        T = np.eye(4)
        
        # Convert mm -> meters
        T[:3, 3] = np.array(self.position_mm) / 1000.0
        
        # Convert degrees -> radians -> rotation matrix
        rpy_rad = np.deg2rad(self.orientation_deg)
        r, p, y = rpy_rad
        
        # Create rotation matrices
        Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
        Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
        
        T[:3, :3] = Rz @ Ry @ Rx
        return T

@dataclass
class RobotWaypoint:
    """Robot waypoint with both joint and TCP data in real-world units."""
    joints_deg: List[float]      # [q1, q2, q3, q4, q5, q6] in degrees
    tcp_position_mm: List[float] # [x, y, z] in mm
    tcp_rotation_deg: List[float] # [rx, ry, rz] in degrees
    
    def move_joint_command(self) -> str:
        """Generate move_joint() command string."""
        j = self.joints_deg
        return f"robot.move_joint({j[0]:.2f}, {j[1]:.2f}, {j[2]:.2f}, {j[3]:.2f}, {j[4]:.2f}, {j[5]:.2f})"
    
    def move_tcp_command(self) -> str:
        """Generate move_tcp() command string."""
        pos, rot = self.tcp_position_mm, self.tcp_rotation_deg
        return f"robot.move_tcp({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}, {rot[0]:.1f}, {rot[1]:.1f}, {rot[2]:.1f})"
    
    def blend_point_command(self, radius_mm: float = 5.0) -> str:
        """Generate blend_point() command string."""
        j = self.joints_deg
        return f"robot.blend_point([{j[0]:.2f}, {j[1]:.2f}, {j[2]:.2f}, {j[3]:.2f}, {j[4]:.2f}, {j[5]:.2f}], {radius_mm})"

@dataclass
class RobotMotionPlan:
    """Complete robot motion plan ready for execution."""
    waypoints: List[RobotWaypoint]
    execution_time_sec: float
    planning_time_ms: float
    success: bool
    error_message: str = ""
    
    def generate_robot_program(self, speed_percent: float = 25.0, blend_radius_mm: float = 5.0) -> str:
        """Generate complete robot program."""
        if not self.success or not self.waypoints:
            return f"# ERROR: {self.error_message}"
        
        lines = [
            f"# Robot motion program - {len(self.waypoints)} waypoints",
            f"# Execution time: {self.execution_time_sec:.2f} seconds",
            f"# Planning time: {self.planning_time_ms:.1f} ms",
            f"",
            f"robot.set_speed({speed_percent})",
            f""
        ]
        
        for i, wp in enumerate(self.waypoints):
            if i == 0:
                lines.append(f"# Waypoint {i+1} - Start")
                lines.append(wp.move_joint_command())
            elif i == len(self.waypoints) - 1:
                lines.append(f"# Waypoint {i+1} - End")
                lines.append(wp.move_joint_command())
            else:
                lines.append(f"# Waypoint {i+1} - Blend")
                lines.append(wp.blend_point_command(blend_radius_mm))
            
            # Add TCP info as comment
            tcp = wp.tcp_position_mm + wp.tcp_rotation_deg
            lines.append(f"#   TCP: [{tcp[0]:.1f}, {tcp[1]:.1f}, {tcp[2]:.1f}] mm, [{tcp[3]:.1f}, {tcp[4]:.1f}, {tcp[5]:.1f}]°")
            lines.append("")
        
        return "\\n".join(lines)

class CleanRobotMotionPlanner:
    """
    Clean, consolidated robot motion planner.
    
    Features:
    - Input: mm positions, degree rotations/joints
    - Output: robot API commands ready for execution
    - Internal: automatic unit conversion to meters/radians
    - Compatible: JSON robot data format
    """
    
    def __init__(self):
        """Initialize motion planning system."""
        self.fk = ForwardKinematics()
        self.ik = InverseKinematics(self.fk)
        self.motion_planner = MotionPlanner(self.fk, self.ik)
    
    def get_current_pose_from_joints(self, joints_deg: List[float]) -> RobotPose:
        """Get current TCP pose from joint angles."""
        # Convert to radians and compute forward kinematics
        q_rad = np.deg2rad(joints_deg)
        T = self.fk.compute_forward_kinematics(q_rad)
        
        # Extract position (convert m -> mm)
        position_mm = (T[:3, 3] * 1000.0).tolist()
        
        # Extract orientation (convert rotation matrix -> RPY -> degrees)
        R = T[:3, :3]
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        
        if sy > 1e-6:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
        
        orientation_deg = np.rad2deg([x, y, z]).tolist()
        
        return RobotPose(position_mm, orientation_deg)
    
    def plan_motion(self, 
                   current_joints_deg: List[float],
                   target_position_mm: List[float],
                   target_orientation_deg: List[float]) -> RobotMotionPlan:
        """
        Plan robot motion from current joints to target pose.
        
        Args:
            current_joints_deg: Current joint angles [q1, q2, q3, q4, q5, q6] in degrees
            target_position_mm: Target position [x, y, z] in mm  
            target_orientation_deg: Target orientation [rx, ry, rz] in degrees
            
        Returns:
            RobotMotionPlan ready for robot execution
        """
        start_time_ms = 0  # Will be set from result
        
        try:
            # Create target pose and convert to transformation matrix
            target_pose = RobotPose(target_position_mm, target_orientation_deg)
            T_target = target_pose.to_transformation_matrix()
            
            # Get current pose
            current_pose = self.get_current_pose_from_joints(current_joints_deg)
            T_current = current_pose.to_transformation_matrix()
            
            # Convert target pose to joint configuration using IK
            q_target, ik_success = self.ik.solve(T_target)
            if not ik_success:
                return RobotMotionPlan(
                    waypoints=[], 
                    execution_time_sec=0.0,
                    planning_time_ms=0.0,
                    success=False, 
                    error_message="IK failed for target pose"
                )
            
            # Get current joint configuration
            q_current = np.deg2rad(current_joints_deg)
            
            # Plan motion using joint space planning
            result = self.motion_planner.plan_motion(
                q_current, q_target,
                strategy=PlanningStrategy.JOINT_SPACE,
                waypoint_count=5
            )
            
            start_time_ms = result.planning_time * 1000
            
            if result.status.value != 'success':
                return RobotMotionPlan(
                    waypoints=[], 
                    execution_time_sec=0.0,
                    planning_time_ms=start_time_ms,
                    success=False, 
                    error_message=result.error_message or f"Planning failed: {result.status.value}"
                )
            
            # Convert waypoints to robot format
            waypoints = []
            if result.plan and result.plan.joint_waypoints:
                for q_rad in result.plan.joint_waypoints:
                    # Convert joint angles to degrees
                    joints_deg = np.rad2deg(q_rad).tolist()
                    
                    # Get TCP pose for this joint configuration  
                    tcp_pose = self.get_current_pose_from_joints(joints_deg)
                    
                    waypoint = RobotWaypoint(
                        joints_deg=joints_deg,
                        tcp_position_mm=tcp_pose.position_mm,
                        tcp_rotation_deg=tcp_pose.orientation_deg
                    )
                    waypoints.append(waypoint)
            
            return RobotMotionPlan(
                waypoints=waypoints,
                execution_time_sec=getattr(result.plan, 'total_time', 0.0) if result.plan else 0.0,
                planning_time_ms=start_time_ms,
                success=True
            )
            
        except Exception as e:
            return RobotMotionPlan(
                waypoints=[],
                execution_time_sec=0.0, 
                planning_time_ms=start_time_ms,
                success=False,
                error_message=f"Exception: {str(e)}"
            )
    
    def load_robot_data_from_json(self, json_file: str) -> List[Dict]:
        """Load robot waypoints from JSON file in real-world units."""
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            waypoints = []
            for wp in data.get('waypoints', []):
                joint_pos_rad = wp.get('joint_positions', [])
                tcp_pos = wp.get('tcp_position', [])
                
                if len(joint_pos_rad) >= 6 and len(tcp_pos) >= 6:
                    waypoint = {
                        'timestamp': wp.get('timestamp', 0),
                        'joints_deg': np.rad2deg(joint_pos_rad[:6]).tolist(),
                        'position_mm': tcp_pos[:3],  # Already in mm
                        'rotation_deg': np.rad2deg(tcp_pos[3:6]).tolist()
                    }
                    waypoints.append(waypoint)
            
            return waypoints
            
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return []

def demo():
    """Demonstrate clean robot interface."""
    print("🤖 CLEAN ROBOT MOTION PLANNER")
    print("=" * 32)
    
    # Initialize
    planner = CleanRobotMotionPlanner()
    print("✅ System initialized")
    
    # Current robot state
    current_joints = [0.0, -30.0, 45.0, 0.0, -15.0, 0.0]  # degrees
    target_pos = [300.0, 200.0, 400.0]  # mm
    target_rot = [0.0, 0.0, 90.0]       # degrees
    
    print(f"\\n📍 MOTION PLANNING:")
    print(f"From joints: {current_joints} degrees")
    print(f"To position: {target_pos} mm")  
    print(f"To orientation: {target_rot} degrees")
    
    # Plan motion
    plan = planner.plan_motion(current_joints, target_pos, target_rot)
    
    if plan.success:
        print(f"\\n✅ SUCCESS!")
        print(f"Waypoints: {len(plan.waypoints)}")
        print(f"Planning time: {plan.planning_time_ms:.1f} ms")
        print(f"Execution time: {plan.execution_time_sec:.2f} seconds")
        
        print(f"\\n🔧 ROBOT API COMMANDS:")
        print("robot.set_speed(25.0)")
        for i, wp in enumerate(plan.waypoints[:3]):
            if i == 0:
                print(f"# Start")
                print(wp.move_joint_command())
            else:
                print(f"# Waypoint {i+1}")
                print(wp.blend_point_command())
        
        print(f"\\n📋 GENERATED PROGRAM:")
        program = plan.generate_robot_program()
        print(program[:300] + "..." if len(program) > 300 else program)
        
    else:
        print(f"❌ FAILED: {plan.error_message}")
    
    print(f"\\n✅ READY FOR ROBOT EXECUTION!")

if __name__ == "__main__":
    demo()
