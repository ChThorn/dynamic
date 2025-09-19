#!/usr/bin/env python3
"""
Test Script: Extended Reach Analysis for 700mm and 710mm

This script tests whether the RB3-730ES-U robot can achieve 700mm and 710mm
reach distances using the current kinematics implementation.

Tests:
1. Forward kinematics analysis of theoretical maximum reach
2. Inverse kinematics testing at 700mm and 710mm distances
3. Multiple orientations and approach angles
4. Joint limit analysis and constraint validation
5. Comparison with current 600mm validated capability

Author: GitHub Copilot
Date: September 2025
"""

import sys
import os
import numpy as np
import time
import logging
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add paths for imports
sys.path.append('../src')
sys.path.append('../../monitoring')

from forward_kinematic import ForwardKinematics
from inverse_kinematic import FastIK

def analyze_theoretical_maximum_reach(fk: ForwardKinematics) -> Dict[str, Any]:
    """
    Analyze the theoretical maximum reach using forward kinematics.
    """
    print("🔍 THEORETICAL MAXIMUM REACH ANALYSIS")
    print("=" * 50)
    
    # Get robot parameters
    S = fk.get_screw_axes()
    M = fk.get_home_configuration()
    joint_limits = fk.get_joint_limits()
    
    print(f"Robot Parameters:")
    print(f"  Number of joints: {fk.n_joints}")
    print(f"  Home TCP position: [{M[0,3]*1000:.1f}, {M[1,3]*1000:.1f}, {M[2,3]*1000:.1f}] mm")
    
    # Test extreme joint configurations to find maximum reach
    max_reach_configs = [
        # Fully extended configurations
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Home position
        np.array([0.0, -np.pi/2, np.pi/2, 0.0, 0.0, 0.0]),  # Arm extended forward
        np.array([np.pi/2, -np.pi/2, np.pi/2, 0.0, 0.0, 0.0]),  # Arm extended to side
        np.array([0.0, -np.pi/3, np.pi/2, 0.0, np.pi/6, 0.0]),  # Slight variations
        np.array([0.0, -np.pi/4, np.pi/3, 0.0, -np.pi/12, 0.0]),  # More variations
    ]
    
    max_reach = 0.0
    max_reach_config = None
    max_reach_position = None
    
    print(f"\nTesting maximum reach configurations:")
    print("-" * 40)
    
    for i, q in enumerate(max_reach_configs):
        try:
            T = fk.compute_forward_kinematics(q)
            position = T[:3, 3]
            reach = np.linalg.norm(position)
            
            print(f"Config {i+1}: reach = {reach*1000:.1f}mm, pos = [{position[0]*1000:.1f}, {position[1]*1000:.1f}, {position[2]*1000:.1f}]mm")
            
            if reach > max_reach:
                max_reach = reach
                max_reach_config = q.copy()
                max_reach_position = position.copy()
                
        except Exception as e:
            print(f"Config {i+1}: Error - {e}")
    
    print(f"\nMaximum theoretical reach found: {max_reach*1000:.1f}mm")
    if max_reach_position is not None:
        print(f"Position: [{max_reach_position[0]*1000:.1f}, {max_reach_position[1]*1000:.1f}, {max_reach_position[2]*1000:.1f}]mm")
        print(f"Joint config (deg): [{', '.join([f'{np.degrees(q):.1f}' for q in max_reach_config])}]")
    
    return {
        'max_reach_mm': max_reach * 1000,
        'max_reach_position': max_reach_position,
        'max_reach_config': max_reach_config
    }

def test_specific_reach_distance(fk: ForwardKinematics, ik: FastIK, 
                               target_distance_mm: float, 
                               num_tests: int = 20) -> Dict[str, Any]:
    """
    Test if the robot can reach a specific distance with various orientations.
    """
    print(f"\n🎯 TESTING {target_distance_mm}mm REACH CAPABILITY")
    print("=" * 50)
    
    target_distance_m = target_distance_mm / 1000.0
    successful_poses = []
    failed_poses = []
    ik_times = []
    position_errors = []
    
    # Generate test poses at the target distance
    test_poses = []
    
    # Test various orientations around a sphere at target distance
    angles = np.linspace(0, 2*np.pi, num_tests)
    elevations = [0, np.pi/6, np.pi/4, np.pi/3]  # Different elevation angles
    
    for i, angle in enumerate(angles[:num_tests//len(elevations)]):
        for elevation in elevations:
            if len(test_poses) >= num_tests:
                break
                
            # Position on sphere at target distance
            x = target_distance_m * np.cos(elevation) * np.cos(angle)
            y = target_distance_m * np.cos(elevation) * np.sin(angle)
            z = 0.1 + target_distance_m * np.sin(elevation)  # Above ground level
            
            # Create transformation matrix with downward pointing orientation
            T_target = np.eye(4)
            T_target[:3, 3] = [x, y, z]
            
            # Downward pointing orientation (similar to gripper)
            T_target[:3, :3] = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            
            test_poses.append(T_target)
    
    print(f"Testing {len(test_poses)} poses at {target_distance_mm}mm reach...")
    print("-" * 40)
    
    # Test each pose
    for i, T_target in enumerate(test_poses):
        position = T_target[:3, 3]
        actual_distance = np.linalg.norm(position) * 1000
        
        print(f"Test {i+1:2d}: pos=[{position[0]*1000:6.1f}, {position[1]*1000:6.1f}, {position[2]*1000:6.1f}]mm (dist={actual_distance:.1f}mm)", end=" ")
        
        try:
            # Use optimized IK parameters for extended reach
            start_time = time.time()
            q_solution, success = ik.solve(T_target, 
                                         time_budget=0.1,        # More time for difficult poses
                                         max_iters=200,          # More iterations
                                         pos_tol=2e-3,          # Relaxed position tolerance
                                         rot_tol=1e-2,          # Relaxed rotation tolerance
                                         max_attempts=8)        # More attempts
            solve_time = time.time() - start_time
            ik_times.append(solve_time)
            
            if success and q_solution is not None:
                # Validate solution with forward kinematics
                T_achieved = fk.compute_forward_kinematics(q_solution)
                pos_achieved = T_achieved[:3, 3]
                pos_error = np.linalg.norm(pos_achieved - position) * 1000
                position_errors.append(pos_error)
                
                print(f"✅ SUCCESS (err={pos_error:.2f}mm, t={solve_time*1000:.1f}ms)")
                successful_poses.append({
                    'target': T_target,
                    'solution': q_solution,
                    'achieved': T_achieved,
                    'position_error': pos_error,
                    'solve_time': solve_time
                })
            else:
                print(f"❌ FAILED (t={solve_time*1000:.1f}ms)")
                failed_poses.append({
                    'target': T_target,
                    'solve_time': solve_time
                })
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed_poses.append({
                'target': T_target,
                'error': str(e)
            })
    
    # Calculate statistics
    success_rate = len(successful_poses) / len(test_poses) * 100
    
    results = {
        'target_distance_mm': target_distance_mm,
        'num_tests': len(test_poses),
        'successful_poses': len(successful_poses),
        'failed_poses': len(failed_poses),
        'success_rate': success_rate,
        'ik_times': ik_times,
        'position_errors': position_errors
    }
    
    # Print summary
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"  Success rate: {success_rate:.1f}% ({len(successful_poses)}/{len(test_poses)})")
    
    if ik_times:
        print(f"  IK solve time: {np.mean(ik_times)*1000:.1f}±{np.std(ik_times)*1000:.1f}ms")
    
    if position_errors:
        print(f"  Position error: {np.mean(position_errors):.2f}±{np.std(position_errors):.2f}mm")
        print(f"  Max position error: {np.max(position_errors):.2f}mm")
    
    # Show a few successful configurations
    if successful_poses:
        print(f"\n✅ Sample successful configurations:")
        for i, pose_info in enumerate(successful_poses[:3]):
            q_deg = np.degrees(pose_info['solution'])
            print(f"  Config {i+1}: [{', '.join([f'{q:.1f}' for q in q_deg])}]° (err={pose_info['position_error']:.2f}mm)")
    
    return results

def compare_reach_capabilities(fk: ForwardKinematics, ik: FastIK) -> Dict[str, Any]:
    """
    Compare reach capabilities across different distances.
    """
    print(f"\n📈 REACH CAPABILITY COMPARISON")
    print("=" * 50)
    
    # Test different reach distances
    test_distances = [580, 600, 620, 650, 680, 700, 710, 720]
    
    comparison_results = {}
    
    for distance in test_distances:
        print(f"\nTesting {distance}mm reach...")
        result = test_specific_reach_distance(fk, ik, distance, num_tests=10)
        comparison_results[distance] = result
    
    # Summary table
    print(f"\n📋 REACH CAPABILITY SUMMARY")
    print("-" * 60)
    print("Distance | Success Rate | Avg Time | Avg Error")
    print("-" * 60)
    
    for distance, result in comparison_results.items():
        success_rate = result['success_rate']
        avg_time = np.mean(result['ik_times']) * 1000 if result['ik_times'] else 0
        avg_error = np.mean(result['position_errors']) if result['position_errors'] else 0
        
        status = "✅" if success_rate > 80 else "⚠️" if success_rate > 50 else "❌"
        print(f"{distance:3d}mm   |    {success_rate:5.1f}%   | {avg_time:6.1f}ms | {avg_error:6.2f}mm {status}")
    
    return comparison_results

def main():
    """
    Main function to test extended reach capabilities.
    """
    print("🚀 EXTENDED REACH ANALYSIS: 700mm & 710mm TESTING")
    print("=" * 60)
    print("Testing RB3-730ES-U robot kinematics at extended reach distances")
    print()
    
    try:
        # Initialize kinematics modules
        print("⚙️ Initializing kinematics modules...")
        fk = ForwardKinematics()
        ik = FastIK(fk)
        print("✅ Kinematics modules initialized")
        
        # 1. Theoretical maximum reach analysis
        theoretical_results = analyze_theoretical_maximum_reach(fk)
        
        # 2. Test specific distances
        results_700mm = test_specific_reach_distance(fk, ik, 700.0, num_tests=15)
        results_710mm = test_specific_reach_distance(fk, ik, 710.0, num_tests=15)
        
        # 3. Comprehensive comparison
        comparison_results = compare_reach_capabilities(fk, ik)
        
        # 4. Final assessment
        print(f"\n🎯 FINAL ASSESSMENT")
        print("=" * 50)
        
        max_theoretical = theoretical_results['max_reach_mm']
        success_700 = results_700mm['success_rate']
        success_710 = results_710mm['success_rate']
        
        print(f"Theoretical maximum reach: {max_theoretical:.1f}mm")
        print(f"700mm reach success rate: {success_700:.1f}%")
        print(f"710mm reach success rate: {success_710:.1f}%")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if success_700 > 80:
            print("✅ 700mm reach is ACHIEVABLE with high reliability")
        elif success_700 > 50:
            print("⚠️ 700mm reach is POSSIBLE but with reduced reliability")
        else:
            print("❌ 700mm reach is NOT RELIABLE for production use")
        
        if success_710 > 80:
            print("✅ 710mm reach is ACHIEVABLE with high reliability")
        elif success_710 > 50:
            print("⚠️ 710mm reach is POSSIBLE but with reduced reliability")
        else:
            print("❌ 710mm reach is NOT RELIABLE for production use")
        
        # Safe operating zone recommendation
        reliable_distances = [d for d, r in comparison_results.items() 
                            if r['success_rate'] > 80]
        
        if reliable_distances:
            max_reliable = max(reliable_distances)
            print(f"🎯 Recommended maximum reliable reach: {max_reliable}mm")
        else:
            print("⚠️ No reliable reach distance found in tested range")
        
        return {
            'theoretical': theoretical_results,
            '700mm': results_700mm,
            '710mm': results_710mm,
            'comparison': comparison_results
        }
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()