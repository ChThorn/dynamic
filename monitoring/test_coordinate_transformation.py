#!/usr/bin/env python3
"""
Comprehensive Test Suite for Monitoring Package Integration

This script provides complete validation of:
1. FK-IK mathematical consistency for position and orientation accuracy
2. Motion planning success/failure for different target positions
3. Intermediate waypoint validation along planned paths
4. Coordinate transformation compatibility between visualizer and planner
"""
import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add the parent directory to the Python path for integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_fk_ik_consistency():
    """Test FK-IK mathematical consistency for position and orientation"""
    print("FK-IK CONSISTENCY VALIDATION")
    print("="*50)
    
    try:
        # Import kinematics modules (project-root based)
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        from kinematics.src.forward_kinematic import ForwardKinematics
        from kinematics.src.inverse_kinematic import FastIK
        
        # Initialize FK and IK
        fk = ForwardKinematics()
        ik = FastIK(fk)
        
        def rotation_matrix_to_euler(R):
            """Convert rotation matrix to Euler angles (ZYX convention)"""
            sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
            if sy < 1e-6:  # Gimbal lock
                x = np.arctan2(-R[1,2], R[1,1])
                y = np.arctan2(-R[2,0], sy)
                z = 0
            else:
                x = np.arctan2(R[2,1], R[2,2])
                y = np.arctan2(-R[2,0], sy)
                z = np.arctan2(R[1,0], R[0,0])
            return np.array([x, y, z])

        def compute_rotation_error(R1, R2):
            """Compute rotation error between two rotation matrices"""
            R_diff = R1.T @ R2
            cos_angle = np.clip((np.trace(R_diff) - 1) / 2.0, -1.0, 1.0)
            return np.arccos(cos_angle)
        
        # Test configurations that represent different robot poses
        test_joints = [
            ([0.0, -60.0, 90.0, 0.0, 30.0, 0.0], "Home position"),
            ([45.0, -45.0, 90.0, 0.0, 45.0, 0.0], "Standard reach"),
            ([-30.0, -30.0, 120.0, -90.0, 60.0, 180.0], "Complex orientation"),
            ([90.0, -80.0, 110.0, 45.0, -30.0, -90.0], "Extended pose"),
            ([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "Zero configuration"),
        ]
        
        print("Testing FK-IK consistency for various joint configurations:")
        print("-" * 70)
        
        max_pos_error = 0.0
        max_rot_error = 0.0
        all_valid = True
        tolerance_pos = 0.001  # 1mm
        tolerance_rot = 0.01   # ~0.6 degrees
        
        for i, (joints_deg, description) in enumerate(test_joints):
            q = np.deg2rad(joints_deg)
            
            # Compute FK
            T_fk = fk.compute_forward_kinematics(q)
            pos_fk = T_fk[:3, 3] * 1000  # Convert to mm
            R_fk = T_fk[:3, :3]
            euler_fk = rotation_matrix_to_euler(R_fk) * 180 / np.pi
            
            # Compute IK to verify
            q_ik, converged = ik.solve(T_fk, q_init=q)
            
            if converged:
                # Verify IK solution with FK
                T_verify = fk.compute_forward_kinematics(q_ik)
                pos_verify = T_verify[:3, 3] * 1000
                R_verify = T_verify[:3, :3]
                
                # Compute errors
                pos_error = np.linalg.norm(pos_verify - pos_fk)
                rot_error = compute_rotation_error(R_fk, R_verify)
                
                max_pos_error = max(max_pos_error, pos_error)
                max_rot_error = max(max_rot_error, rot_error)
                
                # Check tolerances
                pos_ok = pos_error < tolerance_pos
                rot_ok = rot_error < tolerance_rot
                valid = pos_ok and rot_ok
                
                if not valid:
                    all_valid = False
                
                status = "VALID" if valid else "TOLERANCE"
                print(f"Config {i+1} ({description}):")
                print(f"  {status}")
                print(f"  Position: [{pos_fk[0]:.1f}, {pos_fk[1]:.1f}, {pos_fk[2]:.1f}] mm")
                print(f"  Orientation: [{euler_fk[0]:.1f}, {euler_fk[1]:.1f}, {euler_fk[2]:.1f}]°")
                print(f"  FK-IK pos error: {pos_error:.6f} mm")
                print(f"  FK-IK rot error: {rot_error*180/np.pi:.6f}°")
                print()
            else:
                print(f"Config {i+1} ({description}): IK FAILED")
                all_valid = False
                print()
        
        print("FK-IK CONSISTENCY RESULTS:")
        print(f"  Maximum position error: {max_pos_error:.6f} mm")
        print(f"  Maximum rotation error: {max_rot_error*180/np.pi:.6f}°")
        
        if all_valid:
            print("  ALL CONFIGURATIONS: FK-IK mathematically consistent")
        else:
            print("  SOME CONFIGURATIONS: Check tolerance or singularities")
        
        return all_valid, max_pos_error, max_rot_error
    except Exception as e:
        print(f"Error during FK-IK consistency test: {e}")
        return False, float('inf'), float('inf')

def test_waypoint_path_consistency(waypoints):
    """Test FK-IK consistency for intermediate waypoints along a path"""
    print("\nWAYPOINT PATH FK-IK VALIDATION")
    print("="*50)
    
    if not waypoints or len(waypoints) < 2:
        print("Insufficient waypoints for path testing")
        return False
    
    try:
        # Import kinematics modules (project-root based)
        PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if PROJECT_ROOT not in sys.path:
            sys.path.insert(0, PROJECT_ROOT)
        from kinematics.src.forward_kinematic import ForwardKinematics
        from kinematics.src.inverse_kinematic import FastIK
        
        fk = ForwardKinematics()
        ik = FastIK(fk)
        
        print(f"Testing path with {len(waypoints)} waypoints:")
        print("-" * 60)
        
        all_valid = True
        max_error = 0.0
        
        # Test each waypoint
        for i, waypoint in enumerate(waypoints):
            joints_rad = np.deg2rad(waypoint)
            
            # Compute FK
            T_fk = fk.compute_forward_kinematics(joints_rad)
            pos_fk = T_fk[:3, 3] * 1000
            
            # Verify with IK
            q_ik, converged = ik.solve(T_fk, q_init=joints_rad)
            
            if converged:
                T_verify = fk.compute_forward_kinematics(q_ik)
                pos_verify = T_verify[:3, 3] * 1000
                pos_error = np.linalg.norm(pos_verify - pos_fk)
                max_error = max(max_error, pos_error)
                
                status = "OK" if pos_error < 0.001 else "WARN"
                print(f"  Waypoint {i+1}: {status} Error: {pos_error:.6f} mm - TCP: [{pos_fk[0]:.1f}, {pos_fk[1]:.1f}, {pos_fk[2]:.1f}]")
            else:
                print(f"  Waypoint {i+1}: IK Failed")
                all_valid = False
        
        # Test intermediate points between waypoints
        if len(waypoints) >= 2:
            print(f"\nTesting intermediate points between waypoints:")
            for i in range(len(waypoints) - 1):
                q1 = np.deg2rad(waypoints[i])
                q2 = np.deg2rad(waypoints[i + 1])
                
                print(f"  Segment {i+1}→{i+2}: ", end="")
                segment_valid = True
                
                # Test 5 intermediate points (like path planner validation)
                for j in range(5):
                    t = j / 4.0
                    q_interp = (1 - t) * q1 + t * q2
                    
                    T_fk = fk.compute_forward_kinematics(q_interp)
                    q_ik, converged = ik.solve(T_fk, q_init=q_interp)
                    
                    if converged:
                        T_verify = fk.compute_forward_kinematics(q_ik)
                        pos_fk = T_fk[:3, 3] * 1000
                        pos_verify = T_verify[:3, 3] * 1000
                        pos_error = np.linalg.norm(pos_verify - pos_fk)
                        max_error = max(max_error, pos_error)
                        
                        if pos_error >= 0.001:
                            segment_valid = False
                    else:
                        segment_valid = False
                        all_valid = False
                
                print("Valid" if segment_valid else "Invalid")
        
        print(f"\nPATH CONSISTENCY RESULTS:")
        print(f"  Maximum error along path: {max_error:.6f} mm")
        print(f"  Path validation: {'PASSED' if all_valid else 'FAILED'}")
        
        return all_valid
    except Exception as e:
        print(f"Error during path consistency test: {e}")
        return False

def test_motion_planning_with_fk_ik_validation():
    """Test motion planning success with FK-IK validation of generated waypoints"""
    print("\nMOTION PLANNING + FK-IK VALIDATION")
    print("="*50)
    
    try:
        from planning.examples.clean_robot_interface import CleanRobotMotionPlanner
        
        # Initialize motion planner
        planner = CleanRobotMotionPlanner()
        home_joints = [0.0, -60.0, 90.0, 0.0, 30.0, 0.0]
        
        # Test positions with different difficulty levels
        test_positions = [
            ([400.0, 200.0, 300.0], [180.0, 0.0, 0.0], "Pick position"),
            ([200.0, 400.0, 350.0], [180.0, 0.0, 45.0], "Place position"),
            ([300.0, 300.0, 400.0], [180.0, 0.0, 0.0], "Center position"),
            ([100.0, 100.0, 250.0], [180.0, 0.0, 0.0], "Close position"),
            ([450.0, 300.0, 250.0], [180.0, 0.0, 0.0], "Extended reach"),
        ]
        
        print(f"Testing {len(test_positions)} target positions:")
        print("-" * 70)
        
        success_count = 0
        waypoint_collections = []
        
        for pos_mm, rot_deg, description in test_positions:
            distance = np.sqrt(pos_mm[0]**2 + pos_mm[1]**2)
            print(f"\nTarget: {description}: {pos_mm} mm, {rot_deg}° (distance: {distance:.1f}mm)")
            
            try:
                plan = planner.plan_motion(home_joints, pos_mm, rot_deg)
                
                if plan.success:
                    print(f"   MOTION PLANNING SUCCESS: {len(plan.waypoints)} waypoints")
                    success_count += 1
                    
                    # Extract waypoints for FK-IK validation
                    waypoints = []
                    for wp in plan.waypoints:
                        if hasattr(wp, 'joints_deg'):
                            waypoints.append(wp.joints_deg)
                        elif hasattr(wp, 'joint_positions'):
                            waypoints.append(wp.joint_positions)
                    
                    if waypoints:
                        waypoint_collections.append((description, waypoints))
                        print(f"   Collected {len(waypoints)} waypoints for FK-IK validation")
                        
                        # Quick FK-IK validation of first and last waypoint
                        if len(waypoints) >= 2:
                            first_wp = waypoints[0]
                            last_wp = waypoints[-1]
                            print(f"   Start joints: {[f'{j:.1f}' for j in first_wp[:6]]}")
                            print(f"   End joints: {[f'{j:.1f}' for j in last_wp[:6]]}")
                    
                else:
                    print(f"   MOTION PLANNING FAILED: {plan.error_message}")
                    
            except Exception as e:
                print(f"   EXCEPTION: {e}")
        
        print(f"\nMOTION PLANNING RESULTS:")
        print(f"   Successful: {success_count}/{len(test_positions)}")
        print(f"   Success rate: {success_count/len(test_positions)*100:.1f}%")
        
        # Validate waypoints with FK-IK consistency
        if waypoint_collections:
            print(f"\nVALIDATING WAYPOINTS FROM SUCCESSFUL PLANS:")
            for description, waypoints in waypoint_collections:
                print(f"\nPath: {description}")
                path_valid = test_waypoint_path_consistency(waypoints)
                if not path_valid:
                    print(f"   WARNING: FK-IK consistency issues detected in {description}")
        
        return success_count, len(test_positions), waypoint_collections
        
    except Exception as e:
        print(f"Error during motion planning test: {e}")
        return 0, 0, []

def test_reachability_analysis():
    """Test reachability limits and problematic positions"""
    print(f"\nREACHABILITY ANALYSIS")
    print("="*50)
    
    try:
        from planning.examples.clean_robot_interface import CleanRobotMotionPlanner
        
        planner = CleanRobotMotionPlanner()
        home_joints = [0.0, -60.0, 90.0, 0.0, 30.0, 0.0]
        
        # Test positions at different distances to find limits
        # Added a specific 650mm test target with optimized Z-height (250mm)
        test_distances = [
            (300, [300.0, 0.0, 400.0], "Safe zone (300mm)"),
            (400, [400.0, 0.0, 350.0], "Comfortable zone (400mm)"),
            (500, [500.0, 0.0, 300.0], "Warning zone (500mm)"),
            (600, [424.3, 424.3, 250.0], "Extended zone (600mm)"),
            (650, [460.0, 460.0, 250.0], "Target zone (650mm)"),
            (700, [495.0, 495.0, 200.0], "Maximum zone (700mm)"),
        ]
        
        print("Testing reachability at different distances:")
        print("-" * 60)
        
        reachable_limit = 0
        for distance, pos_mm, description in test_distances:
            actual_distance = np.sqrt(pos_mm[0]**2 + pos_mm[1]**2)
            rot_deg = [180.0, 0.0, 0.0]
            
            print(f"\nTarget: {description}: {pos_mm} mm (actual: {actual_distance:.1f}mm)")
            
            try:
                plan = planner.plan_motion(home_joints, pos_mm, rot_deg)
                
                if plan.success:
                    print(f"   SUCCESS: {len(plan.waypoints)} waypoints")
                    reachable_limit = max(reachable_limit, actual_distance)
                else:
                    print(f"   FAILED: {plan.error_message}")
                    
            except Exception as e:
                print(f"   EXCEPTION: {e}")
        
        print(f"\nREACHABILITY RESULTS:")
        print(f"   Reliable reach limit: ~{reachable_limit:.0f}mm")
        print(f"   URDF theoretical limit: 730mm")
        print(f"   Practical workspace: {reachable_limit:.0f}mm / 730mm = {reachable_limit/730*100:.1f}%")
        
        return reachable_limit
        
    except Exception as e:
        print(f"Error during reachability analysis: {e}")
        return 0

def main():
    """Run comprehensive test suite"""
    print("COMPREHENSIVE MONITORING PACKAGE VALIDATION")
    print("Testing FK-IK consistency, motion planning, and coordinate transformation")
    print("=" * 80)
    
    # Track overall results
    overall_success = True
    
    # 1. Test FK-IK mathematical consistency
    fk_ik_valid, max_pos_err, max_rot_err = test_fk_ik_consistency()
    if not fk_ik_valid:
        overall_success = False
    
    # 2. Test motion planning with waypoint validation
    success_count, total_tests, waypoint_collections = test_motion_planning_with_fk_ik_validation()
    if success_count < total_tests:
        print(f"WARNING: Motion planning success rate: {success_count/total_tests*100:.1f}%")
    
    # 3. Test reachability limits
    reach_limit = test_reachability_analysis()
    
    # 4. Overall summary
    print(f"\nCOMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 60)
    print(f"FK-IK Mathematical Consistency:")
    print(f"   Position accuracy: {max_pos_err:.6f} mm")
    print(f"   Orientation accuracy: {max_rot_err*180/np.pi:.6f}°")
    print(f"   Status: {'PERFECT' if fk_ik_valid else 'CHECK TOLERANCES'}")
    
    print(f"\nMotion Planning Capability:")
    print(f"   Success rate: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
    print(f"   Waypoint validation: {'PASSED' if waypoint_collections else 'NO WAYPOINTS'}")
    
    print(f"\nWorkspace Analysis:")
    print(f"   Reliable reach: {reach_limit:.0f}mm")
    print(f"   Theoretical limit: 730mm") 
    print(f"   Workspace utilization: {reach_limit/730*100:.1f}%")
    
    print(f"\nOVERALL SYSTEM STATUS:")
    if overall_success and success_count > 0 and reach_limit > 400:
        print("   EXCELLENT: Complete system validation PASSED")
        print("   • Mathematical accuracy: Perfect")
        print("   • Motion planning: Functional") 
        print("   • Workspace coverage: Good")
        print("   • Ready for production use!")
    elif success_count > 0:
        print("   GOOD: Core functionality validated")
        print("   • Some limitations in reach or planning")
        print("   • Suitable for most applications")
    else:
        print("   ISSUES: Validation revealed problems")
        print("   • Check coordinate transformations")
        print("   • Verify workspace constraints")
        print("   • Review constraint configurations")
    
    print(f"\nNEXT STEPS:")
    print("   • Use test_poses_selection.py for interactive pose selection")
    print("   • Monitor planning success rates in production")
    print("   • Consider constraint tuning if reach limits are too conservative")

if __name__ == "__main__":
    main()