#!/usr/bin/env python3
"""
Robot Parameter Analysis Script

Analyzes the screw axes and home configuration of the RB3-730ES-U robot
to understand its geometric structure and identify opportunities for
analytical inverse kinematics solutions.
"""

import numpy as np
import sys
import os

# Add the kinematics package to path
sys.path.append('/home/ubuntu/kinematics/src')

from forward_kinematic import ForwardKinematics
import json

def analyze_screw_axes(S):
    """Analyze the screw axes matrix to understand robot structure."""
    print("=== SCREW AXES ANALYSIS ===")
    print(f"Screw axes matrix S (6x{S.shape[1]}):")
    print(S)
    print()
    
    # Analyze each joint
    for i in range(S.shape[1]):
        print(f"Joint {i+1}:")
        omega = S[:3, i]  # Angular velocity
        v = S[3:, i]      # Linear velocity
        
        print(f"  ω = {omega}")
        print(f"  v = {v}")
        print(f"  |ω| = {np.linalg.norm(omega):.6f}")
        print(f"  |v| = {np.linalg.norm(v):.6f}")
        
        # Check if it's a revolute joint (|ω| = 1, v ≠ 0) or prismatic (ω = 0, |v| = 1)
        if np.allclose(np.linalg.norm(omega), 1.0, atol=1e-6):
            print(f"  Type: Revolute joint")
            # For revolute joints, find the axis of rotation
            if not np.allclose(omega, 0):
                print(f"  Rotation axis: {omega / np.linalg.norm(omega)}")
        elif np.allclose(omega, 0, atol=1e-6):
            print(f"  Type: Prismatic joint")
            print(f"  Translation direction: {v / np.linalg.norm(v)}")
        else:
            print(f"  Type: General screw motion")
        print()

def analyze_home_configuration(M):
    """Analyze the home configuration matrix."""
    print("=== HOME CONFIGURATION ANALYSIS ===")
    print("Home configuration matrix M:")
    print(M)
    print()
    
    R = M[:3, :3]  # Rotation part
    p = M[:3, 3]   # Translation part
    
    print(f"Home position: {p}")
    print(f"Home orientation (rotation matrix):")
    print(R)
    
    # Convert to RPY for easier interpretation
    def matrix_to_rpy(R):
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2,1], R[2,2])
            y = np.arctan2(-R[2,0], sy)
            z = np.arctan2(R[1,0], R[0,0])
        else:
            x = np.arctan2(-R[1,2], R[1,1])
            y = np.arctan2(-R[2,0], sy)
            z = 0
            
        return np.array([x, y, z])
    
    rpy = matrix_to_rpy(R)
    print(f"Home orientation (RPY): {np.rad2deg(rpy)} degrees")
    print()

def check_spherical_wrist(S):
    """Check if the robot has a spherical wrist (last 3 joints intersect at a point)."""
    print("=== SPHERICAL WRIST ANALYSIS ===")
    
    # For a spherical wrist, the last 3 joint axes should intersect at a common point
    # This means the screw axes for joints 4, 5, 6 should have their ω vectors
    # intersecting at the wrist center
    
    if S.shape[1] < 6:
        print("Robot has less than 6 joints, spherical wrist analysis not applicable")
        return False
    
    # Get the last 3 joints (joints 4, 5, 6)
    omega4, omega5, omega6 = S[:3, 3], S[:3, 4], S[:3, 5]
    v4, v5, v6 = S[3:, 3], S[3:, 4], S[3:, 5]
    
    print("Last 3 joints (potential wrist joints):")
    print(f"Joint 4: ω = {omega4}, v = {v4}")
    print(f"Joint 5: ω = {omega5}, v = {v5}")
    print(f"Joint 6: ω = {omega6}, v = {v6}")
    
    # Check if joints 4 and 6 have parallel axes (common for spherical wrists)
    if not np.allclose(omega4, 0) and not np.allclose(omega6, 0):
        omega4_unit = omega4 / np.linalg.norm(omega4)
        omega6_unit = omega6 / np.linalg.norm(omega6)
        
        # Check if parallel or anti-parallel
        dot_product = np.dot(omega4_unit, omega6_unit)
        if np.abs(dot_product) > 0.99:  # Nearly parallel
            print(f"Joints 4 and 6 are nearly parallel (dot product: {dot_product:.6f})")
            print("This suggests a spherical wrist configuration!")
            return True
    
    # Check if joint axes intersect (more complex geometric analysis)
    # For now, we'll use a simpler heuristic
    print("Detailed intersection analysis would require more complex geometry")
    return False

def analyze_robot_geometry():
    """Analyze the overall robot geometry and structure."""
    print("=== ROBOT GEOMETRY ANALYSIS ===")
    
    # Initialize forward kinematics
    fk = ForwardKinematics()
    S = fk.get_screw_axes()
    M = fk.get_home_configuration()
    
    print(f"Robot has {S.shape[1]} joints")
    print(f"All joints appear to be revolute based on screw axes")
    
    # Analyze joint structure
    print("\nJoint axis directions:")
    for i in range(S.shape[1]):
        omega = S[:3, i]
        if not np.allclose(omega, 0):
            axis = omega / np.linalg.norm(omega)
            print(f"Joint {i+1}: {axis}")
    
    # Check for common robot configurations
    print("\nChecking for common robot configurations:")
    
    # Check if first 3 joints form a positioning mechanism
    omega1, omega2, omega3 = S[:3, 0], S[:3, 1], S[:3, 2]
    print(f"First 3 joint axes:")
    print(f"  J1: {omega1}")
    print(f"  J2: {omega2}")  
    print(f"  J3: {omega3}")
    
    return fk, S, M

def test_forward_kinematics_with_data():
    """Test forward kinematics with actual robot data."""
    print("=== FORWARD KINEMATICS VALIDATION ===")
    
    # Load the JSON data
    with open('third_20250710_162459.json', 'r') as f:
        data = json.load(f)
    
    fk = ForwardKinematics()
    
    # Test with a few waypoints
    waypoints = data['waypoints'][:5]  # First 5 waypoints
    
    print("Testing forward kinematics with recorded waypoints:")
    for i, wp in enumerate(waypoints):
        q_deg = np.array(wp['joint_positions'])
        q = np.deg2rad(q_deg)  # Convert from degrees to radians
        tcp_recorded = np.array(wp['tcp_position'])
        
        # Compute forward kinematics
        T = fk.compute_forward_kinematics(q)
        tcp_computed = T[:3, 3] * 1000  # Convert to mm
        
        # Extract orientation (assuming RPY format in recorded data)
        R = T[:3, :3]
        rpy_computed = fk.matrix_to_rpy(R)
        
        print(f"\nWaypoint {i+1}:")
        print(f"  Joint angles: {q_deg} degrees")
        print(f"  Recorded TCP: {tcp_recorded[:3]} mm")
        print(f"  Computed TCP: {tcp_computed} mm")
        print(f"  Position error: {np.linalg.norm(tcp_computed - tcp_recorded[:3]):.3f} mm")

if __name__ == "__main__":
    print("Robot Parameter Analysis for RB3-730ES-U")
    print("=" * 50)
    
    # Analyze robot structure
    fk, S, M = analyze_robot_geometry()
    
    # Detailed analysis
    analyze_screw_axes(S)
    analyze_home_configuration(M)
    
    # Check for spherical wrist
    has_spherical_wrist = check_spherical_wrist(S)
    
    # Test with real data
    test_forward_kinematics_with_data()
    
    print("\n=== SUMMARY ===")
    print(f"Robot type: 6-DOF serial manipulator")
    print(f"All joints: Revolute")
    print(f"Spherical wrist: {'Yes' if has_spherical_wrist else 'Needs further analysis'}")
    print(f"Forward kinematics: Implemented using Product of Exponentials")
    print(f"Suitable for analytical IK: {'Yes' if has_spherical_wrist else 'Requires geometric analysis'}")

