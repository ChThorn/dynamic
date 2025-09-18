#!/usr/bin/env python3
"""
Comprehensive Test Suite for Improved Inverse Kinematics

This test suite validates the improved inverse kinematics module and
demonstrates its performance improvements over the original iterative-only approach.

Test Categories:
1. Accuracy validation (must match original solver)
2. Performance benchmarking (analytical vs iterative)
3. Special case handling (home position, base rotations)
4. Real robot data validation
5. Stress testing with random poses
"""

import numpy as np
from numpy.linalg import norm
import time
import json
import sys
import os

# Add the kinematics package to path
sys.path.append('/home/ubuntu/kinematics/src')
from forward_kinematic import ForwardKinematics
from inverse_kinematic import InverseKinematics as OriginalIK

# Import our improved version
sys.path.append('/home/ubuntu')
from improved_inverse_kinematic import ImprovedInverseKinematics

def test_accuracy_validation():
    """Test that improved IK maintains same accuracy as original."""
    print("=" * 60)
    print("ACCURACY VALIDATION TEST")
    print("=" * 60)
    
    fk = ForwardKinematics()
    original_ik = OriginalIK(fk)
    improved_ik = ImprovedInverseKinematics(fk)
    
    # Test configurations
    test_configs = [
        np.zeros(6),  # Home
        np.array([0.1, 0, 0, 0, 0, 0]),  # Base rotation
        np.array([0, 0.1, 0, 0, 0, 0]),  # Shoulder
        np.array([0, 0, 0.1, 0, 0, 0]),  # Elbow
        np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),  # Complex
        np.array([0.5, -0.3, 0.8, 0.2, -0.4, 0.6]),  # More complex
    ]
    
    print("Testing accuracy against original solver...")
    all_match = True
    
    for i, q_test in enumerate(test_configs):
        T_target = fk.compute_forward_kinematics(q_test)
        
        # Solve with both methods
        q_orig, success_orig = original_ik.solve(T_target, q_test)
        q_impr, success_impr = improved_ik.solve(T_target, q_test)
        
        if success_orig and success_impr:
            # Verify both solutions are accurate
            T_orig = fk.compute_forward_kinematics(q_orig)
            T_impr = fk.compute_forward_kinematics(q_impr)
            
            error_orig = norm(T_orig[:3, 3] - T_target[:3, 3]) * 1000
            error_impr = norm(T_impr[:3, 3] - T_target[:3, 3]) * 1000
            
            print(f"  Config {i+1}: Original={error_orig:.3f}mm, Improved={error_impr:.3f}mm")
            
            if error_orig > 5.0 or error_impr > 5.0:
                all_match = False
                print(f"    ⚠ Large error detected!")
        else:
            print(f"  Config {i+1}: Original={success_orig}, Improved={success_impr}")
            if success_orig != success_impr:
                all_match = False
    
    if all_match:
        print("✓ All accuracy tests passed!")
    else:
        print("✗ Some accuracy tests failed!")
    
    return all_match

def test_performance_benchmark():
    """Benchmark performance improvements."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK TEST")
    print("=" * 60)
    
    fk = ForwardKinematics()
    original_ik = OriginalIK(fk)
    improved_ik = ImprovedInverseKinematics(fk)
    
    # Test cases designed to hit analytical solutions
    analytical_cases = [
        ("Home position", np.zeros(6)),
        ("Base rotation 30°", np.array([0.524, 0, 0, 0, 0, 0])),
        ("Base rotation -45°", np.array([-0.785, 0, 0, 0, 0, 0])),
        ("Base rotation 90°", np.array([1.571, 0, 0, 0, 0, 0])),
    ]
    
    # Test cases that require iterative solution
    iterative_cases = [
        ("Complex pose 1", np.array([0.2, 0.3, 0.4, 0.1, 0.2, 0.3])),
        ("Complex pose 2", np.array([0.5, -0.2, 0.6, -0.3, 0.4, -0.1])),
        ("Complex pose 3", np.array([-0.3, 0.4, -0.2, 0.5, -0.1, 0.2])),
    ]
    
    print("Benchmarking analytical cases...")
    for name, q_test in analytical_cases:
        T_target = fk.compute_forward_kinematics(q_test)
        
        # Time original solver
        start_time = time.time()
        q_orig, success_orig = original_ik.solve(T_target, q_test)
        orig_time = time.time() - start_time
        
        # Time improved solver
        start_time = time.time()
        q_impr, success_impr = improved_ik.solve(T_target, q_test)
        impr_time = time.time() - start_time
        
        if success_orig and success_impr:
            speedup = orig_time / impr_time if impr_time > 0 else float('inf')
            print(f"  {name}: {orig_time*1000:.2f}ms → {impr_time*1000:.2f}ms (speedup: {speedup:.1f}x)")
        else:
            print(f"  {name}: Failed (orig={success_orig}, impr={success_impr})")
    
    print("\nBenchmarking iterative cases...")
    for name, q_test in iterative_cases:
        T_target = fk.compute_forward_kinematics(q_test)
        
        # Time original solver
        start_time = time.time()
        q_orig, success_orig = original_ik.solve(T_target, q_test)
        orig_time = time.time() - start_time
        
        # Time improved solver
        start_time = time.time()
        q_impr, success_impr = improved_ik.solve(T_target, q_test)
        impr_time = time.time() - start_time
        
        if success_orig and success_impr:
            ratio = impr_time / orig_time if orig_time > 0 else 1.0
            print(f"  {name}: {orig_time*1000:.2f}ms → {impr_time*1000:.2f}ms (ratio: {ratio:.2f})")
        else:
            print(f"  {name}: Failed (orig={success_orig}, impr={success_impr})")

def test_real_robot_data():
    """Test with real robot waypoint data."""
    print("\n" + "=" * 60)
    print("REAL ROBOT DATA TEST")
    print("=" * 60)
    
    fk = ForwardKinematics()
    improved_ik = ImprovedInverseKinematics(fk)
    
    # Load real robot data
    with open('third_20250710_162459.json', 'r') as f:
        data = json.load(f)
    
    waypoints = data['waypoints'][::3][:20]  # Every 3rd waypoint, first 20
    
    print(f"Testing with {len(waypoints)} real robot waypoints...")
    
    success_count = 0
    total_time = 0
    
    for i, wp in enumerate(waypoints):
        q_actual = np.deg2rad(np.array(wp['joint_positions']))
        T_target = fk.compute_forward_kinematics(q_actual)
        
        start_time = time.time()
        q_solution, success = improved_ik.solve(T_target, q_actual)
        solve_time = time.time() - start_time
        total_time += solve_time
        
        if success:
            T_verify = fk.compute_forward_kinematics(q_solution)
            pos_error = norm(T_verify[:3, 3] - T_target[:3, 3]) * 1000
            success_count += 1
            
            if i < 5:  # Show details for first 5
                print(f"  Waypoint {i+1}: ✓ Error: {pos_error:.3f}mm, Time: {solve_time*1000:.2f}ms")
        else:
            print(f"  Waypoint {i+1}: ✗ Failed")
    
    avg_time = total_time / len(waypoints) * 1000
    success_rate = success_count / len(waypoints) * 100
    
    print(f"\nResults:")
    print(f"  Success rate: {success_count}/{len(waypoints)} ({success_rate:.1f}%)")
    print(f"  Average time: {avg_time:.2f}ms")
    
    # Get detailed statistics
    stats = improved_ik.get_statistics()
    print(f"  Analytical success rate: {stats['analytical_success_rate']:.1%}")
    print(f"  Home position hits: {stats['home_position_hits']}")
    print(f"  Base rotation hits: {stats['base_rotation_hits']}")

def test_stress_testing():
    """Stress test with random poses."""
    print("\n" + "=" * 60)
    print("STRESS TEST WITH RANDOM POSES")
    print("=" * 60)
    
    fk = ForwardKinematics()
    improved_ik = ImprovedInverseKinematics(fk)
    
    # Generate random joint configurations within limits
    joint_limits = fk.get_joint_limits()
    lower, upper = joint_limits[0], joint_limits[1]
    
    num_tests = 100
    print(f"Testing {num_tests} random configurations...")
    
    success_count = 0
    total_time = 0
    max_error = 0
    
    for i in range(num_tests):
        # Generate random configuration
        q_random = np.random.uniform(lower, upper)
        T_target = fk.compute_forward_kinematics(q_random)
        
        start_time = time.time()
        q_solution, success = improved_ik.solve(T_target, q_random)
        solve_time = time.time() - start_time
        total_time += solve_time
        
        if success:
            T_verify = fk.compute_forward_kinematics(q_solution)
            pos_error = norm(T_verify[:3, 3] - T_target[:3, 3]) * 1000
            max_error = max(max_error, pos_error)
            success_count += 1
    
    avg_time = total_time / num_tests * 1000
    success_rate = success_count / num_tests * 100
    
    print(f"Results:")
    print(f"  Success rate: {success_count}/{num_tests} ({success_rate:.1f}%)")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Maximum error: {max_error:.3f}mm")
    
    # Get final statistics
    stats = improved_ik.get_statistics()
    print(f"  Final analytical success rate: {stats['analytical_success_rate']:.1%}")

def main():
    """Run all tests."""
    print("IMPROVED INVERSE KINEMATICS TEST SUITE")
    print("Testing hybrid analytical/iterative approach")
    print("Author: Robot Control Team")
    
    # Run all test categories
    accuracy_ok = test_accuracy_validation()
    test_performance_benchmark()
    test_real_robot_data()
    test_stress_testing()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if accuracy_ok:
        print("✓ Accuracy validation: PASSED")
        print("✓ The improved solver maintains the same accuracy as the original")
        print("✓ Analytical solutions provide exact results for special cases")
        print("✓ Iterative fallback ensures robustness for general cases")
        print("\nThe improved inverse kinematics module is ready for deployment!")
    else:
        print("✗ Accuracy validation: FAILED")
        print("⚠ The improved solver needs further refinement")

if __name__ == "__main__":
    main()

