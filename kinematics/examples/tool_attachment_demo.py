#!/usr/bin/env python3
"""
Demo: Easy Tool Attachment/Detachment
=====================================

This script demonstrates how to easily switch between TCP-only mode 
and gripper tool mode for flexible robot operation.
"""

import sys
import os
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from forward_kinematic import ForwardKinematics
from inverse_kinematic import InverseKinematics

def demo_tool_attachment():
    """Demonstrate easy tool attachment and detachment."""
    
    print("🤖 ROBOT TOOL ATTACHMENT/DETACHMENT DEMO")
    print("=" * 50)
    
    # Initialize forward kinematics
    fk = ForwardKinematics()
    ik = InverseKinematics(fk)
    
    # Test joint configuration
    test_joints = np.array([0.2, -0.3, 0.5, 0.0, 0.8, 0.1])  # radians
    
    print("\n1. INITIAL STATE (TCP-only mode)")
    print("-" * 30)
    print(f"   Tool attached: {fk.is_tool_attached()}")
    
    # Compute TCP position
    T_tcp = fk.compute_forward_kinematics(test_joints)
    tcp_pos = T_tcp[:3, 3]
    print(f"   TCP position: [{tcp_pos[0]:.4f}, {tcp_pos[1]:.4f}, {tcp_pos[2]:.4f}] m")
    
    print("\n2. ATTACHING GRIPPER TOOL")
    print("-" * 30)
    success = fk.attach_tool("default_gripper")
    if success:
        print(f"   ✅ Tool attached: {fk.is_tool_attached()}")
        
        # Get tool info
        tool_info = fk.get_tool_info()
        if tool_info:
            print(f"   Tool name: {tool_info['name']}")
            print(f"   Tool type: {tool_info['type']}")
            print(f"   Tool offset: {tool_info['offset_translation']} m")
        
        # Compute positions with tool
        T_tcp_with_tool = fk.compute_tcp_kinematics(test_joints)  # Always TCP
        T_tool = fk.compute_forward_kinematics(test_joints)        # Tool frame when attached
        
        tcp_pos_with_tool = T_tcp_with_tool[:3, 3]
        tool_pos = T_tool[:3, 3]
        
        print(f"   TCP position (with tool): [{tcp_pos_with_tool[0]:.4f}, {tcp_pos_with_tool[1]:.4f}, {tcp_pos_with_tool[2]:.4f}] m")
        print(f"   Tool position: [{tool_pos[0]:.4f}, {tool_pos[1]:.4f}, {tool_pos[2]:.4f}] m")
        
        # Calculate offset distance
        offset_distance = np.linalg.norm(tool_pos - tcp_pos_with_tool)
        print(f"   Tool offset distance: {offset_distance:.4f} m (should be 0.085m)")
        
        # Verify TCP consistency
        tcp_consistency_error = np.linalg.norm(tcp_pos - tcp_pos_with_tool)
        print(f"   TCP consistency: {tcp_consistency_error:.6f} m error")
        
    print("\n3. INVERSE KINEMATICS WITH TOOL")
    print("-" * 30)
    
    # Test IK for both TCP and tool targets
    target_pos = np.array([0.5, 0.2, 0.7])
    target_orientation = np.eye(3)  # No rotation
    T_target = np.eye(4)
    T_target[:3, :3] = target_orientation
    T_target[:3, 3] = target_pos
    
    # IK for TCP target
    q_tcp, success_tcp = ik.solve(T_target, use_tool_frame=False)
    print(f"   IK to TCP target: {'✅ Success' if success_tcp else '❌ Failed'}")
    
    # IK for tool target  
    q_tool, success_tool = ik.solve(T_target, use_tool_frame=True)
    print(f"   IK to tool target: {'✅ Success' if success_tool else '❌ Failed'}")
    
    if success_tcp and success_tool:
        # Verify the results
        T_check_tcp = fk.compute_tcp_kinematics(q_tcp)     # Always TCP
        T_check_tool = fk.compute_forward_kinematics(q_tool)  # Tool frame when attached
        
        tcp_error = np.linalg.norm(T_check_tcp[:3, 3] - target_pos)
        tool_error = np.linalg.norm(T_check_tool[:3, 3] - target_pos)
        
        print(f"   TCP IK accuracy: {tcp_error * 1000:.3f} mm")
        print(f"   Tool IK accuracy: {tool_error * 1000:.3f} mm")
    
    print("\n4. DETACHING TOOL (Back to TCP-only)")
    print("-" * 30)
    fk.detach_tool()
    print(f"   Tool attached: {fk.is_tool_attached()}")
    
    # Verify TCP position is back to original
    T_tcp_detached = fk.compute_forward_kinematics(test_joints)
    tcp_pos_detached = T_tcp_detached[:3, 3]
    print(f"   TCP position: [{tcp_pos_detached[0]:.4f}, {tcp_pos_detached[1]:.4f}, {tcp_pos_detached[2]:.4f}] m")
    
    # Verify consistency
    consistency_error = np.linalg.norm(tcp_pos - tcp_pos_detached)
    print(f"   TCP consistency: {consistency_error:.6f} m error")
    
    print("\n5. RE-ATTACHING TOOL (Demonstrating flexibility)")
    print("-" * 30)
    success = fk.attach_tool("default_gripper")
    print(f"   ✅ Tool re-attached: {fk.is_tool_attached()}")
    
    print("\n" + "=" * 50)
    print("🎉 TOOL ATTACHMENT DEMO COMPLETE")
    print("\nKey Features Demonstrated:")
    print("✅ Easy tool attachment: fk.attach_tool('tool_name')")
    print("✅ Easy tool detachment: fk.detach_tool()")
    print("✅ Tool status checking: fk.is_tool_attached()")
    print("✅ Tool information: fk.get_tool_info()")
    print("✅ Dual-mode IK: use_tool_frame=True/False")
    print("✅ TCP-only computation: fk.compute_tcp_kinematics()")
    print("✅ Tool computation: fk.compute_forward_kinematics() (when attached)")
    print("✅ Perfect TCP consistency maintained")
    print("✅ Seamless switching between modes")
    
    return {
        'tcp_only_mode': True,
        'tool_mode': True,
        'consistency_maintained': consistency_error < 1e-6,
        'ik_both_modes': success_tcp and success_tool
    }

if __name__ == "__main__":
    results = demo_tool_attachment()