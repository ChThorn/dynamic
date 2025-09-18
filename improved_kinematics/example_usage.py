#!/usr/bin/env python3
"""
Example Usage of Improved Inverse Kinematics

This example demonstrates how to use the improved inverse kinematics solver
as a drop-in replacement for the original iterative solver.
"""

import numpy as np
from forward_kinematic import ForwardKinematics
from improved_inverse_kinematic import ImprovedInverseKinematics

def main():
    print("Improved Inverse Kinematics - Usage Example")
    print("=" * 50)
    
    # Initialize forward and inverse kinematics
    fk = ForwardKinematics()
    ik = ImprovedInverseKinematics(fk)
    
    print("1. Basic Usage - Same as Original")
    print("-" * 30)
    
    # Define a target pose (example: slight base rotation)
    q_target = np.array([0.5, 0, 0, 0, 0, 0])  # 30 degree base rotation
    T_target = fk.compute_forward_kinematics(q_target)
    
    print(f"Target joint angles: {np.rad2deg(q_target)} degrees")
    print(f"Target position: {T_target[:3, 3] * 1000} mm")
    
    # Solve inverse kinematics
    q_solution, success = ik.solve(T_target, q_target)
    
    if success:
        print(f"Solution found: {np.rad2deg(q_solution)} degrees")
        
        # Verify the solution
        T_verify = fk.compute_forward_kinematics(q_solution)
        pos_error = np.linalg.norm(T_verify[:3, 3] - T_target[:3, 3]) * 1000
        print(f"Position error: {pos_error:.3f} mm")
    else:
        print("No solution found")
    
    print("\n2. Performance Comparison")
    print("-" * 30)
    
    # Test different types of poses
    test_cases = [
        ("Home position", np.zeros(6)),
        ("Base rotation", np.array([0.3, 0, 0, 0, 0, 0])),
        ("Complex pose", np.array([0.2, 0.3, 0.4, 0.1, 0.2, 0.3]))
    ]
    
    import time
    
    for name, q_test in test_cases:
        T_test = fk.compute_forward_kinematics(q_test)
        
        start_time = time.time()
        q_sol, success = ik.solve(T_test, q_test)
        solve_time = time.time() - start_time
        
        print(f"{name}: {solve_time*1000:.2f}ms ({'✓' if success else '✗'})")
    
    print("\n3. Performance Statistics")
    print("-" * 30)
    
    stats = ik.get_statistics()
    print(f"Total calls: {stats['total_calls']}")
    print(f"Analytical success rate: {stats['analytical_success_rate']:.1%}")
    print(f"Home position hits: {stats['home_position_hits']}")
    print(f"Base rotation hits: {stats['base_rotation_hits']}")
    print(f"Average solve time: {stats['average_time']*1000:.2f}ms")
    
    print("\n4. Tool Frame Support (if available)")
    print("-" * 30)
    
    # Example with tool frame (same interface as original)
    if hasattr(fk, 'tool') and fk.tool:
        q_tool, success = ik.solve_tool_pose(T_target, q_target)
        print(f"Tool pose solution: {'✓' if success else '✗'}")
    else:
        print("No tool attached - using TCP frame")
        q_tcp, success = ik.solve_tcp_pose(T_target, q_target)
        print(f"TCP pose solution: {'✓' if success else '✗'}")
    
    print("\nThe improved solver provides the same interface as the original")
    print("while offering significant performance improvements for common poses!")

if __name__ == "__main__":
    main()

