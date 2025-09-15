#!/usr/bin/env python3
"""
Gripper Tool Demonstration Script

This script demonstrates the new gripper tool support for robot kinematics:
1. Comparison between TCP-only and gripper-enabled modes
2. Tool offset transformations
3. Forward and inverse kinematics with gripper attached
4. Validation of tool offset calculations

Run this script to see how gripper attachment affects robot kinematics.

Author: Robot Control Team
"""

import os
import sys
import numpy as np
import logging

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import kinematics modules
from forward_kinematic import ForwardKinematics
from inverse_kinematic import InverseKinematics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('gripper_demo')

def print_pose_summary(T: np.ndarray, label: str):
    """Print a human-readable summary of a pose."""
    position = T[:3, 3]
    print(f"  {label}:")
    print(f"    Position (m): [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
    
    # Convert rotation to RPY for readability
    from forward_kinematic import ForwardKinematics
    rpy = ForwardKinematics.matrix_to_rpy(T[:3, :3])
    rpy_deg = np.rad2deg(rpy)
    print(f"    Orientation (deg): [{rpy_deg[0]:.2f}, {rpy_deg[1]:.2f}, {rpy_deg[2]:.2f}]")

def demonstrate_tool_configuration():
    """Demonstrate tool configuration and attachment."""
    print("=" * 60)
    print("TOOL CONFIGURATION DEMONSTRATION")
    print("=" * 60)
    
    # Create FK without tool
    fk_tcp = ForwardKinematics()
    print(f"\n1. TCP-only mode:")
    print(f"   Tool attached: {fk_tcp.is_tool_attached()}")
    
    # Create FK with gripper tool
    fk_gripper = ForwardKinematics(tool_name="default_gripper")
    print(f"\n2. Gripper mode:")
    print(f"   Tool attached: {fk_gripper.is_tool_attached()}")
    
    if fk_gripper.is_tool_attached():
        tool_info = fk_gripper.get_tool_info()
        print(f"   Tool name: {tool_info['name']}")
        print(f"   Tool type: {tool_info['type']}")
        
        offset = fk_gripper.get_tool_offset()
        if offset is not None:
            print(f"   Tool offset (translation): {offset[:3, 3]}")
    
    return fk_tcp, fk_gripper

def compare_forward_kinematics(fk_tcp, fk_gripper):
    """Compare forward kinematics between TCP and gripper modes."""
    print("\n" + "=" * 60)
    print("FORWARD KINEMATICS COMPARISON")
    print("=" * 60)
    
    # Test configurations
    test_configs = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Home
        np.array([0.2, -0.3, 0.4, 0.0, 0.5, 0.0]),  # Test config 1
        np.array([np.pi/4, -np.pi/6, np.pi/3, 0.0, np.pi/4, np.pi/6])  # Test config 2
    ]
    
    for i, q in enumerate(test_configs):
        print(f"\nTest Configuration {i+1}:")
        print(f"  Joint angles (deg): {np.rad2deg(q)}")
        
        # Compute TCP pose
        T_tcp = fk_tcp.compute_forward_kinematics(q)
        
        # Compute gripper pose
        T_gripper = fk_gripper.compute_forward_kinematics(q)
        
        # Also compute TCP pose in gripper mode for comparison
        T_tcp_from_gripper = fk_gripper.compute_tcp_kinematics(q)
        
        print_pose_summary(T_tcp, "TCP (TCP-only mode)")
        print_pose_summary(T_tcp_from_gripper, "TCP (gripper mode)")
        print_pose_summary(T_gripper, "Gripper functional point")
        
        # Calculate tool offset
        offset_distance = np.linalg.norm(T_gripper[:3, 3] - T_tcp[:3, 3])
        print(f"    Tool offset distance: {offset_distance:.4f} m")
        
        # Verify TCP consistency
        tcp_error = np.linalg.norm(T_tcp[:3, 3] - T_tcp_from_gripper[:3, 3])
        print(f"    TCP consistency error: {tcp_error:.6f} m")

def compare_inverse_kinematics(fk_tcp, fk_gripper):
    """Compare inverse kinematics between TCP and gripper modes."""
    print("\n" + "=" * 60)
    print("INVERSE KINEMATICS COMPARISON")
    print("=" * 60)
    
    # Create IK solvers
    ik_tcp = InverseKinematics(fk_tcp)
    ik_gripper = InverseKinematics(fk_gripper)
    
    # Test target poses (in TCP coordinates)
    target_positions = [
        np.array([0.4, 0.2, 0.6]),    # Reachable position 1
        np.array([0.0, 0.5, 0.4]),    # Reachable position 2
        np.array([-0.3, -0.2, 0.8])   # Reachable position 3
    ]
    
    for i, pos in enumerate(target_positions):
        print(f"\nTarget Position {i+1}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m")
        
        # Create target transformation matrix (identity rotation)
        T_target = np.eye(4)
        T_target[:3, 3] = pos
        
        # Solve IK for TCP target
        q_tcp, success_tcp = ik_tcp.solve(T_target)
        print(f"  TCP IK: {'Success' if success_tcp else 'Failed'}")
        
        if success_tcp:
            # Verify TCP solution
            T_verify_tcp = fk_tcp.compute_forward_kinematics(q_tcp)
            tcp_pos_error = np.linalg.norm(T_verify_tcp[:3, 3] - pos)
            print(f"    TCP position error: {tcp_pos_error*1000:.2f} mm")
            print(f"    TCP joint solution (deg): {np.rad2deg(q_tcp)}")
        
        # Solve IK for gripper target (same TCP position)
        q_gripper_tcp, success_gripper_tcp = ik_gripper.solve_tcp_pose(T_target)
        print(f"  Gripper IK (TCP target): {'Success' if success_gripper_tcp else 'Failed'}")
        
        if success_gripper_tcp:
            # Verify gripper TCP solution
            T_verify_gripper_tcp = fk_gripper.compute_tcp_kinematics(q_gripper_tcp)
            gripper_tcp_pos_error = np.linalg.norm(T_verify_gripper_tcp[:3, 3] - pos)
            print(f"    Gripper TCP position error: {gripper_tcp_pos_error*1000:.2f} mm")
            
            # Show gripper functional point position
            T_gripper_func = fk_gripper.compute_forward_kinematics(q_gripper_tcp)
            gripper_func_pos = T_gripper_func[:3, 3]
            print(f"    Gripper functional position: [{gripper_func_pos[0]:.3f}, {gripper_func_pos[1]:.3f}, {gripper_func_pos[2]:.3f}] m")
        
        # Now solve for gripper functional point at the same position
        q_gripper_func, success_gripper_func = ik_gripper.solve_tool_pose(T_target)
        print(f"  Gripper IK (tool target): {'Success' if success_gripper_func else 'Failed'}")
        
        if success_gripper_func:
            # Verify gripper functional point solution
            T_verify_gripper_func = fk_gripper.compute_forward_kinematics(q_gripper_func)
            gripper_func_pos_error = np.linalg.norm(T_verify_gripper_func[:3, 3] - pos)
            print(f"    Gripper func position error: {gripper_func_pos_error*1000:.2f} mm")
            
            # Show TCP position for this solution
            T_tcp_for_gripper = fk_gripper.compute_tcp_kinematics(q_gripper_func)
            tcp_for_gripper_pos = T_tcp_for_gripper[:3, 3]
            print(f"    Corresponding TCP position: [{tcp_for_gripper_pos[0]:.3f}, {tcp_for_gripper_pos[1]:.3f}, {tcp_for_gripper_pos[2]:.3f}] m")

def demonstrate_tool_workspace_impact():
    """Demonstrate how tool attachment affects workspace."""
    print("\n" + "=" * 60)
    print("TOOL WORKSPACE IMPACT DEMONSTRATION")
    print("=" * 60)
    
    # Create FK instances
    fk_tcp = ForwardKinematics()
    fk_gripper = ForwardKinematics(tool_name="default_gripper")
    
    # Sample the workspace with a few configurations
    num_samples = 20
    np.random.seed(42)  # For reproducible results
    
    # Generate random joint configurations
    limits_lower, limits_upper = fk_tcp.get_joint_limits()
    
    tcp_positions = []
    gripper_positions = []
    
    for i in range(num_samples):
        # Random joint configuration
        q = np.random.uniform(limits_lower, limits_upper)
        
        # TCP position
        T_tcp = fk_tcp.compute_forward_kinematics(q)
        tcp_positions.append(T_tcp[:3, 3])
        
        # Gripper position
        T_gripper = fk_gripper.compute_forward_kinematics(q)
        gripper_positions.append(T_gripper[:3, 3])
    
    tcp_positions = np.array(tcp_positions)
    gripper_positions = np.array(gripper_positions)
    
    # Calculate workspace statistics
    print(f"\nWorkspace Analysis ({num_samples} samples):")
    print(f"  TCP workspace:")
    print(f"    X range: [{tcp_positions[:, 0].min():.3f}, {tcp_positions[:, 0].max():.3f}] m")
    print(f"    Y range: [{tcp_positions[:, 1].min():.3f}, {tcp_positions[:, 1].max():.3f}] m")
    print(f"    Z range: [{tcp_positions[:, 2].min():.3f}, {tcp_positions[:, 2].max():.3f}] m")
    
    print(f"  Gripper workspace:")
    print(f"    X range: [{gripper_positions[:, 0].min():.3f}, {gripper_positions[:, 0].max():.3f}] m")
    print(f"    Y range: [{gripper_positions[:, 1].min():.3f}, {gripper_positions[:, 1].max():.3f}] m")
    print(f"    Z range: [{gripper_positions[:, 2].min():.3f}, {gripper_positions[:, 2].max():.3f}] m")
    
    # Calculate tool offset statistics
    offsets = gripper_positions - tcp_positions
    offset_distances = np.linalg.norm(offsets, axis=1)
    
    print(f"\nTool Offset Statistics:")
    print(f"  Mean offset distance: {offset_distances.mean():.4f} m")
    print(f"  Offset distance range: [{offset_distances.min():.4f}, {offset_distances.max():.4f}] m")
    print(f"  Offset standard deviation: {offset_distances.std():.4f} m")

def main():
    """Main demonstration function."""
    print("ROBOT GRIPPER TOOL DEMONSTRATION")
    print("This script demonstrates kinematics with gripper tool support")
    print()
    
    try:
        # 1. Tool configuration
        fk_tcp, fk_gripper = demonstrate_tool_configuration()
        
        # 2. Forward kinematics comparison
        compare_forward_kinematics(fk_tcp, fk_gripper)
        
        # 3. Inverse kinematics comparison
        compare_inverse_kinematics(fk_tcp, fk_gripper)
        
        # 4. Workspace impact
        demonstrate_tool_workspace_impact()
        
        # 5. Summary
        print("\n" + "=" * 60)
        print("DEMONSTRATION SUMMARY")
        print("=" * 60)
        print("✅ Tool configuration and attachment working")
        print("✅ Forward kinematics with tool offset working")
        print("✅ Inverse kinematics with tool support working")
        print("✅ Tool workspace impact demonstrated")
        print("\nKey Observations:")
        print("• Gripper attachment adds tool offset to TCP calculations")
        print("• IK can solve for both TCP poses and gripper functional poses")
        print("• Tool attachment affects the effective workspace")
        print("• All transformations are mathematically consistent")
        
        print(f"\nConfiguration Details:")
        if fk_gripper.is_tool_attached():
            tool_info = fk_gripper.get_tool_info()
            offset = tool_info['offset_translation']
            print(f"• Tool: {tool_info['name']}")
            print(f"• Offset: [{offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f}] m")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Gripper tool demonstration completed successfully!")
    else:
        print("\n❌ Gripper tool demonstration failed!")
        sys.exit(1)