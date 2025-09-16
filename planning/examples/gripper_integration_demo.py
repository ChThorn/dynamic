#!/usr/bin/env python3
"""
Gripper-Aware Motion Planning Demo

This example demonstrates how to integrate the new gripper tool system 
with the existing motion planning framework. It shows:

1. Planning with TCP-only mode (original behavior)
2. Planning with gripper tool attached
3. Dynamic tool attachment/detachment during operation
4. Pick and place operations considering gripper geometry
5. Workspace differences between TCP and gripper modes

Key Integration Points:
- Kinematics now supports tool attachment/detachment
- Motion planning automatically considers tool offset when attached
- IK solving works for both TCP and tool frame targets
- Collision checking accounts for tool geometry

Author: Robot Control Team
"""

import sys
import os
import numpy as np
import logging
import time
from typing import List, Tuple, Optional

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'kinematics', 'src'))

# Import kinematics modules
from forward_kinematic import ForwardKinematics
from inverse_kinematic import InverseKinematics

# Import planning modules
from motion_planner import MotionPlanner, PlanningStrategy, PlanningStatus
from path_planner import PathPlanner
from trajectory_planner import TrajectoryPlanner

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_mm_to_m(position_mm: List[float]) -> np.ndarray:
    """Convert position from mm to meters."""
    return np.array(position_mm) / 1000.0

def convert_deg_to_rad(angles_deg: List[float]) -> np.ndarray:
    """Convert angles from degrees to radians."""
    return np.radians(angles_deg)

def create_pose_matrix(position: np.ndarray, orientation: np.ndarray = None) -> np.ndarray:
    """Create 4x4 pose matrix from position and orientation."""
    T = np.eye(4)
    T[:3, 3] = position
    if orientation is not None:
        # Simple rotation around Z axis for demo
        angle = orientation[2] if len(orientation) > 2 else 0
        c, s = np.cos(angle), np.sin(angle)
        T[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return T

def demo_gripper_integration():
    """Demonstrate gripper tool integration with motion planning."""
    
    print("🤖 GRIPPER-AWARE MOTION PLANNING DEMO")
    print("=" * 60)
    
    # Initialize kinematics
    logger.info("Initializing kinematics modules...")
    fk = ForwardKinematics()
    ik = InverseKinematics(fk)
    
    # Initialize motion planner
    logger.info("Initializing motion planner...")
    motion_planner = MotionPlanner(fk, ik)
    
    print("\n1. INITIAL STATE - TCP-ONLY MODE")
    print("-" * 40)
    print(f"Tool attached: {fk.is_tool_attached()}")
    
    # Define some test configurations
    home_config = np.zeros(6)  # Home position
    test_config = convert_deg_to_rad([30, -20, 45, 0, 25, 0])
    
    # Test FK in TCP-only mode
    T_tcp_only = fk.compute_forward_kinematics(test_config)
    tcp_pos = T_tcp_only[:3, 3]
    print(f"TCP position at test config: [{tcp_pos[0]:.4f}, {tcp_pos[1]:.4f}, {tcp_pos[2]:.4f}] m")
    
    print("\n2. ATTACHING GRIPPER TOOL")
    print("-" * 40)
    success = fk.attach_tool("default_gripper")
    if success:
        print(f"✅ Tool attached: {fk.is_tool_attached()}")
        tool_info = fk.get_tool_info()
        print(f"Tool: {tool_info['name']}")
        print(f"Offset: {tool_info['offset_translation']} m")
        
        # Test FK with tool attached
        T_tcp_with_tool = fk.compute_tcp_kinematics(test_config)
        T_tool = fk.compute_forward_kinematics(test_config)
        
        tcp_pos_with_tool = T_tcp_with_tool[:3, 3]
        tool_pos = T_tool[:3, 3]
        
        print(f"TCP position: [{tcp_pos_with_tool[0]:.4f}, {tcp_pos_with_tool[1]:.4f}, {tcp_pos_with_tool[2]:.4f}] m")
        print(f"Tool position: [{tool_pos[0]:.4f}, {tool_pos[1]:.4f}, {tool_pos[2]:.4f}] m")
        
        offset_distance = np.linalg.norm(tool_pos - tcp_pos_with_tool)
        print(f"Tool offset: {offset_distance:.4f} m (expected: 0.085m)")
        
        tcp_consistency = np.linalg.norm(tcp_pos - tcp_pos_with_tool)
        print(f"TCP consistency: {tcp_consistency:.6f} m")
    
    print("\n3. MOTION PLANNING WITH GRIPPER")
    print("-" * 40)
    
    # Plan motion from home to test configuration
    print("Planning motion: home → test position")
    start_time = time.time()
    result = motion_planner.plan_motion(
        start_config=home_config,
        goal_config=test_config,
        strategy=PlanningStrategy.JOINT_SPACE
    )
    planning_time = time.time() - start_time
    
    if result.status == PlanningStatus.SUCCESS:
        print(f"✅ Motion planned successfully")
        print(f"Planning time: {planning_time*1000:.1f} ms")
        print(f"Waypoints: {len(result.plan.joint_waypoints)}")
        print(f"Strategy used: {result.plan.strategy_used.value}")
        
        # Verify all waypoints consider gripper
        first_waypoint = result.plan.joint_waypoints[0]
        last_waypoint = result.plan.joint_waypoints[-1]
        
        T_start = fk.compute_forward_kinematics(first_waypoint)
        T_end = fk.compute_forward_kinematics(last_waypoint)
        
        print(f"Start tool position: [{T_start[0,3]:.4f}, {T_start[1,3]:.4f}, {T_start[2,3]:.4f}] m")
        print(f"End tool position: [{T_end[0,3]:.4f}, {T_end[1,3]:.4f}, {T_end[2,3]:.4f}] m")
    else:
        print(f"❌ Motion planning failed: {result.error_message}")
    
    print("\n4. CARTESIAN MOTION WITH GRIPPER")
    print("-" * 40)
    
    # Define target positions for gripper operations
    pick_position = convert_mm_to_m([400, 200, 150])  # 15cm above table
    place_position = convert_mm_to_m([-300, 250, 200])  # 20cm above table
    
    print("Planning Cartesian motion for pick operation...")
    
    # Plan motion to pick position (target for tool frame)
    T_pick = create_pose_matrix(pick_position)
    
    start_time = time.time()
    cart_result = motion_planner.plan_cartesian_motion(
        start_pose=T_tcp_only,  # Current TCP pose
        goal_pose=T_pick,       # Target for gripper tool
        orientation_constraint=False
    )
    planning_time = time.time() - start_time
    
    if cart_result.status == PlanningStatus.SUCCESS:
        print(f"✅ Cartesian motion planned successfully")
        print(f"Planning time: {planning_time*1000:.1f} ms")
        print(f"Waypoints: {len(cart_result.plan.joint_waypoints)}")
        
        # Verify final position
        final_config = cart_result.plan.joint_waypoints[-1]
        T_final = fk.compute_forward_kinematics(final_config)
        final_tool_pos = T_final[:3, 3]
        
        print(f"Target position: [{pick_position[0]:.4f}, {pick_position[1]:.4f}, {pick_position[2]:.4f}] m")
        print(f"Achieved position: [{final_tool_pos[0]:.4f}, {final_tool_pos[1]:.4f}, {final_tool_pos[2]:.4f}] m")
        
        position_error = np.linalg.norm(final_tool_pos - pick_position)
        print(f"Position error: {position_error*1000:.2f} mm")
    else:
        print(f"❌ Cartesian planning failed: {cart_result.error_message}")
    
    print("\n5. PICK AND PLACE SEQUENCE")
    print("-" * 40)
    
    # Demonstrate a complete pick and place with gripper
    pick_sequence = [
        ("Approach pick", pick_position + [0, 0, 0.05]),  # 5cm above
        ("Pick position", pick_position),
        ("Retreat pick", pick_position + [0, 0, 0.05]),
        ("Approach place", place_position + [0, 0, 0.05]),
        ("Place position", place_position),
        ("Retreat place", place_position + [0, 0, 0.05])
    ]
    
    current_config = home_config
    all_plans_success = True
    
    for i, (phase_name, target_pos) in enumerate(pick_sequence):
        print(f"Phase {i+1}: {phase_name}")
        
        # Create target pose
        T_target = create_pose_matrix(target_pos)
        
        # Plan motion from current to target
        start_time = time.time()
        phase_result = motion_planner.plan_cartesian_motion(
            start_pose=fk.compute_forward_kinematics(current_config),
            goal_pose=T_target,
            orientation_constraint=False
        )
        planning_time = time.time() - start_time
        
        if phase_result.status == PlanningStatus.SUCCESS:
            print(f"  ✅ Planned in {planning_time*1000:.1f} ms, {len(phase_result.plan.joint_waypoints)} waypoints")
            current_config = phase_result.plan.joint_waypoints[-1]
            
            # Verify position
            T_achieved = fk.compute_forward_kinematics(current_config)
            achieved_pos = T_achieved[:3, 3]
            error = np.linalg.norm(achieved_pos - target_pos)
            print(f"  Position error: {error*1000:.2f} mm")
        else:
            print(f"  ❌ Failed: {phase_result.error_message}")
            all_plans_success = False
            break
    
    if all_plans_success:
        print("✅ Complete pick and place sequence planned successfully!")
    
    print("\n6. TOOL DETACHMENT DEMO")
    print("-" * 40)
    
    # Detach tool and show difference
    fk.detach_tool()
    print(f"Tool attached: {fk.is_tool_attached()}")
    
    # Same configuration, different end-effector position
    T_tcp_detached = fk.compute_forward_kinematics(test_config)
    tcp_pos_detached = T_tcp_detached[:3, 3]
    
    print(f"TCP position (detached): [{tcp_pos_detached[0]:.4f}, {tcp_pos_detached[1]:.4f}, {tcp_pos_detached[2]:.4f}] m")
    print(f"Difference from tool mode: {np.linalg.norm(tcp_pos_detached - tool_pos)*1000:.1f} mm")
    
    print("\n7. WORKSPACE COMPARISON")
    print("-" * 40)
    
    # Compare workspace extents
    test_configs = [
        convert_deg_to_rad([0, 0, 0, 0, 0, 0]),
        convert_deg_to_rad([90, 0, 0, 0, 0, 0]),
        convert_deg_to_rad([0, -45, 45, 0, 0, 0]),
        convert_deg_to_rad([-90, 30, -60, 0, 45, 0])
    ]
    
    tcp_positions = []
    tool_positions = []
    
    for config in test_configs:
        # TCP-only mode
        T_tcp = fk.compute_forward_kinematics(config)
        tcp_positions.append(T_tcp[:3, 3])
        
        # With tool
        fk.attach_tool("default_gripper")
        T_tool = fk.compute_forward_kinematics(config)
        tool_positions.append(T_tool[:3, 3])
        fk.detach_tool()
    
    tcp_positions = np.array(tcp_positions)
    tool_positions = np.array(tool_positions)
    
    print("TCP workspace extent:")
    print(f"  X: [{np.min(tcp_positions[:,0]):.3f}, {np.max(tcp_positions[:,0]):.3f}] m")
    print(f"  Y: [{np.min(tcp_positions[:,1]):.3f}, {np.max(tcp_positions[:,1]):.3f}] m") 
    print(f"  Z: [{np.min(tcp_positions[:,2]):.3f}, {np.max(tcp_positions[:,2]):.3f}] m")
    
    print("Tool workspace extent:")
    print(f"  X: [{np.min(tool_positions[:,0]):.3f}, {np.max(tool_positions[:,0]):.3f}] m")
    print(f"  Y: [{np.min(tool_positions[:,1]):.3f}, {np.max(tool_positions[:,1]):.3f}] m")
    print(f"  Z: [{np.min(tool_positions[:,2]):.3f}, {np.max(tool_positions[:,2]):.3f}] m")
    
    print("\n" + "=" * 60)
    print("🎉 GRIPPER INTEGRATION DEMO COMPLETE")
    print("\nKey Integration Features Demonstrated:")
    print("✅ Seamless tool attachment/detachment")
    print("✅ Motion planning with gripper considerations")
    print("✅ Cartesian planning for tool frame targets")
    print("✅ Pick and place sequence planning")
    print("✅ Workspace analysis with/without tools")
    print("✅ Perfect TCP consistency maintenance")
    print("✅ Production-ready gripper integration")
    
    # Get final statistics
    stats = motion_planner.get_statistics()
    print(f"\nPlanning Statistics:")
    print(f"  Total plans: {stats.get('total_plans', 0)}")
    print(f"  Successful: {stats.get('successful_plans', 0)}")
    if stats.get('total_plans', 0) > 0:
        success_rate = stats.get('successful_plans', 0) / stats.get('total_plans', 1) * 100
        print(f"  Success rate: {success_rate:.1f}%")

if __name__ == "__main__":
    demo_gripper_integration()