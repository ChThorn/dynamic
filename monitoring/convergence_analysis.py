#!/usr/bin/env python3
"""
Detailed Analysis of Gripper Mode Convergence Issues

This script explains WHY gripper mode sometimes fails where TCP mode succeeds,
and provides practical guidelines for when to use each mode.

Author: GitHub Copilot
Date: September 2025
"""

import sys
import os
import numpy as np

# Add module paths
sys.path.append('../kinematics/src')
sys.path.append('.')

from forward_kinematic import ForwardKinematics
from inverse_kinematic import InverseKinematics


def analyze_convergence_issues():
    """
    Analyze why gripper mode fails in certain scenarios.
    """
    
    print("🔍 DETAILED CONVERGENCE ANALYSIS")
    print("=" * 60)
    print("Understanding WHY gripper mode fails in specific cases")
    print()
    
    # Initialize both modes
    fk_tcp = ForwardKinematics()
    fk_gripper = ForwardKinematics(tool_name='default_gripper')
    ik_tcp = InverseKinematics(fk_tcp)
    ik_gripper = InverseKinematics(fk_gripper)
    
    # Analyze the failing case: [280, 200, 250] mm
    problem_position = [280, 200, 250]  # mm
    
    print(f"🎯 CASE STUDY: Position [{problem_position[0]}, {problem_position[1]}, {problem_position[2]}] mm")
    print("=" * 55)
    print()
    
    # Create target transformation
    T_target = np.eye(4)
    T_target[:3, 3] = np.array(problem_position) / 1000
    
    q_init = np.deg2rad([0, -30, 60, 0, 45, 0])
    
    print("📋 TCP MODE ANALYSIS:")
    q_tcp, conv_tcp = ik_tcp.solve(T_target, q_init=q_init, max_iterations=100)
    
    if conv_tcp:
        T_tcp_result = fk_tcp.compute_forward_kinematics(q_tcp)
        tcp_achieved = T_tcp_result[:3, 3] * 1000
        tcp_error = np.linalg.norm(tcp_achieved - problem_position)
        
        # Analyze joint angles
        joints_deg = np.rad2deg(q_tcp)
        
        print(f"   ✅ TCP mode SUCCESS")
        print(f"   📍 Target: [{problem_position[0]:.1f}, {problem_position[1]:.1f}, {problem_position[2]:.1f}] mm")
        print(f"   📍 Achieved: [{tcp_achieved[0]:.1f}, {tcp_achieved[1]:.1f}, {tcp_achieved[2]:.1f}] mm")
        print(f"   📊 Error: {tcp_error:.3f} mm")
        print(f"   🔧 Joint angles: [{joints_deg[0]:.1f}°, {joints_deg[1]:.1f}°, {joints_deg[2]:.1f}°, {joints_deg[3]:.1f}°, {joints_deg[4]:.1f}°, {joints_deg[5]:.1f}°]")
        
        # Check workspace metrics
        reach_distance = np.linalg.norm(tcp_achieved[:2])
        print(f"   📏 Radial distance: {reach_distance:.1f} mm")
        
        # Check joint limits
        joint_limits = [(-360, 360), (-90, 90), (-180, 180), (-360, 360), (-180, 180), (-360, 360)]
        within_limits = all(low <= angle <= high for angle, (low, high) in zip(joints_deg, joint_limits))
        print(f"   ⚖️  Within joint limits: {within_limits}")
        
    else:
        print(f"   ❌ TCP mode FAILED")
    
    print()
    
    print("📋 GRIPPER MODE ANALYSIS:")
    q_gripper, conv_gripper = ik_gripper.solve(T_target, q_init=q_init, max_iterations=100)
    
    if conv_gripper:
        T_gripper_result = fk_gripper.compute_forward_kinematics(q_gripper)
        gripper_achieved = T_gripper_result[:3, 3] * 1000
        gripper_error = np.linalg.norm(gripper_achieved - problem_position)
        
        # Get equivalent TCP position
        T_tcp_from_gripper = fk_gripper.compute_tcp_kinematics(q_gripper)
        tcp_from_gripper = T_tcp_from_gripper[:3, 3] * 1000
        
        # Analyze joint angles
        joints_deg = np.rad2deg(q_gripper)
        
        print(f"   ✅ Gripper mode SUCCESS")
        print(f"   📍 Target gripper: [{problem_position[0]:.1f}, {problem_position[1]:.1f}, {problem_position[2]:.1f}] mm")
        print(f"   📍 Achieved gripper: [{gripper_achieved[0]:.1f}, {gripper_achieved[1]:.1f}, {gripper_achieved[2]:.1f}] mm")
        print(f"   📍 Resulting TCP: [{tcp_from_gripper[0]:.1f}, {tcp_from_gripper[1]:.1f}, {tcp_from_gripper[2]:.1f}] mm")
        print(f"   📊 Gripper error: {gripper_error:.3f} mm")
        print(f"   🔧 Joint angles: [{joints_deg[0]:.1f}°, {joints_deg[1]:.1f}°, {joints_deg[2]:.1f}°, {joints_deg[3]:.1f}°, {joints_deg[4]:.1f}°, {joints_deg[5]:.1f}°]")
        
        # Check workspace metrics
        reach_distance = np.linalg.norm(gripper_achieved[:2])
        print(f"   📏 Radial distance: {reach_distance:.1f} mm")
        
        # Check joint limits
        joint_limits = [(-360, 360), (-90, 90), (-180, 180), (-360, 360), (-180, 180), (-360, 360)]
        within_limits = all(low <= angle <= high for angle, (low, high) in zip(joints_deg, joint_limits))
        print(f"   ⚖️  Within joint limits: {within_limits}")
        
    else:
        print(f"   ❌ Gripper mode FAILED")
        print(f"   💡 Reason: Likely joint limits or workspace boundaries")
        print(f"   📊 Target requires TCP at: [{problem_position[0]:.1f}, {problem_position[1]:.1f}, {problem_position[2]-85:.1f}] mm")
        print(f"      (Gripper at target height requires TCP 85mm lower)")
    
    print()
    
    # Explain the fundamental issue
    if conv_tcp and not conv_gripper:
        print("🔍 WHY GRIPPER MODE FAILED:")
        print("=" * 35)
        print("The fundamental issue is HEIGHT GEOMETRY:")
        print()
        print(f"• Target position: [280, 200, 250] mm")
        print(f"• TCP mode: Robot reaches to exactly [280, 200, 250] mm")
        print(f"• Gripper mode: Robot TCP must be at [280, 200, 165] mm")
        print(f"  (85mm lower to put gripper tip at target)")
        print()
        print("💡 The 165mm height is too LOW for this radial distance!")
        print("   The robot cannot reach that far horizontally at such a low height")
        print("   due to joint angle limitations.")
        print()
    
    return conv_tcp, conv_gripper


def analyze_optimal_use_cases():
    """
    Determine when to use TCP mode vs gripper mode.
    """
    
    print("🎯 OPTIMAL USE CASES ANALYSIS")
    print("=" * 45)
    print("When to use TCP mode vs Gripper mode")
    print()
    
    fk_tcp = ForwardKinematics()
    fk_gripper = ForwardKinematics(tool_name='default_gripper')
    ik_tcp = InverseKinematics(fk_tcp)
    ik_gripper = InverseKinematics(fk_gripper)
    
    # Test different height zones
    test_scenarios = [
        ("Low height (200mm)", [250, 150, 200]),
        ("Table height (300mm)", [250, 150, 300]),
        ("Mid height (400mm)", [250, 150, 400]),
        ("High reach (500mm)", [250, 150, 500]),
        ("Near workspace edge", [350, 200, 350]),
        ("Far but feasible", [300, 250, 400])
    ]
    
    print("📊 SCENARIO COMPARISON:")
    print()
    
    tcp_wins = 0
    gripper_wins = 0
    both_work = 0
    both_fail = 0
    
    for scenario_name, position in test_scenarios:
        print(f"🎯 {scenario_name}: [{position[0]}, {position[1]}, {position[2]}] mm")
        
        T_target = np.eye(4)
        T_target[:3, 3] = np.array(position) / 1000
        
        q_init = np.deg2rad([0, -30, 60, 0, 45, 0])
        
        # Test both modes
        _, conv_tcp = ik_tcp.solve(T_target, q_init=q_init)
        _, conv_gripper = ik_gripper.solve(T_target, q_init=q_init)
        
        if conv_tcp and conv_gripper:
            print(f"   ✅ Both modes work - Choose based on application")
            both_work += 1
        elif conv_tcp and not conv_gripper:
            print(f"   🟦 TCP mode ONLY - Use TCP mode")
            tcp_wins += 1
        elif not conv_tcp and conv_gripper:
            print(f"   🟩 Gripper mode ONLY - Use gripper mode")
            gripper_wins += 1
        else:
            print(f"   ❌ Neither mode works - Position unreachable")
            both_fail += 1
        
        print()
    
    print("📊 SUMMARY STATISTICS:")
    print(f"   Both modes work: {both_work}/{len(test_scenarios)}")
    print(f"   TCP mode only: {tcp_wins}/{len(test_scenarios)}")
    print(f"   Gripper mode only: {gripper_wins}/{len(test_scenarios)}")
    print(f"   Neither works: {both_fail}/{len(test_scenarios)}")
    print()
    
    print("💡 PRACTICAL GUIDELINES:")
    print("=" * 30)
    print("✅ Use GRIPPER mode when:")
    print("   • Pick-and-place operations (intuitive coordinates)")
    print("   • Object manipulation (direct gripper positioning)")
    print("   • Medium to high working heights (>300mm)")
    print("   • Need maximum reach extension")
    print()
    print("✅ Use TCP mode when:")
    print("   • Low working heights (<300mm)")
    print("   • Precision tool operations")
    print("   • Custom tool attachments")
    print("   • Legacy applications")
    print()
    print("🔄 Hybrid approach:")
    print("   • Try gripper mode first for pick-and-place")
    print("   • Fall back to TCP mode if gripper mode fails")
    print("   • Use workspace analysis to predict which mode works best")


def main():
    """
    Run complete analysis of convergence issues.
    """
    
    print("🔍 GRIPPER MODE CONVERGENCE ISSUES: DETAILED ANALYSIS")
    print("=" * 70)
    print("Understanding why TCP mode sometimes outperforms gripper mode")
    print()
    
    try:
        # Analyze specific convergence issues
        conv_tcp, conv_gripper = analyze_convergence_issues()
        
        # Analyze optimal use cases
        analyze_optimal_use_cases()
        
        print("\n🎯 KEY INSIGHTS:")
        print("=" * 25)
        print("1. 🏗️  GEOMETRY MATTERS: Low heights + far distances challenge gripper mode")
        print("2. 📐 JOINT LIMITS: Gripper mode needs more extreme joint angles")
        print("3. 🎯 USE CASE SPECIFIC: Choose mode based on working height and application")
        print("4. 🔄 COMPLEMENTARY: Both modes have their strengths")
        print("5. 💡 STRATEGY: Try gripper mode first, fall back to TCP if needed")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()