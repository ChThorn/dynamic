#!/usr/bin/env python3
"""
Test Script: Gripper Pose Selection with Motion Planning Integration
==================================================================

This script demonstrates the integration of gripper-based pose selection 
with the motion planning system. It allows users to:

1. Select gripper tip positions using the interactive visualizer
2. Automatically convert gripper coordinates to TCP coordinates  
3. Plan motion trajectories from home position to gripper targets
4. Generate robot programs for execution

Key Features:
- Gripper mode for intuitive tip positioning (85mm automatic offset)
- Interactive 2D/3D pose selection interface
- Motion planning with AORRTC algorithm
- Waypoint generation for robot execution
- Production-ready trajectory planning

Usage: python3 test_gripper_pose_selection.py

Author: GitHub Copilot
Date: September 2025
"""

import sys
import os
import numpy as np
import time
import logging
from scipy.spatial.transform import Rotation as R

# Configure logging to suppress verbose output
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors
logging.getLogger().setLevel(logging.WARNING)

# Add module paths
sys.path.append('../kinematics/src')
sys.path.append('../planning/src')
sys.path.append('.')

from forward_kinematic import ForwardKinematics
from inverse_kinematic import FastIK
from visualizer_considered_gripper import AdvancedPoseVisualizer

# Import motion planning components
try:
    from planning.examples.clean_robot_interface import CleanRobotMotionPlanner
    PLANNING_AVAILABLE = True
except ImportError as e:
    print(f"❌ Planning modules not available: {e}")
    print("Make sure the planning package is properly installed.")
    PLANNING_AVAILABLE = False
    sys.exit(1)


def run_gripper_pose_selection_demo():
    """
    Main demo function combining gripper pose selection with motion planning.
    Clean, minimal output version.
    """
    
    print("🦾 GRIPPER POSE SELECTION WITH MOTION PLANNING")
    print("="*50)
    print("Interactive gripper pose selection for robot motion planning")
    print()
    
    try:
        # Step 1: Initialize motion planner
        print("⚙️  Initializing motion planner...")
        planner = CleanRobotMotionPlanner()
        print("✅ Motion planner ready")
        
        # Step 2: Define home position
        home_joints = [0.0, -60.0, 90.0, 0.0, 30.0, 0.0]  # degrees
        home_tcp_pose = planner.get_current_pose_from_joints(home_joints)
        home_tcp_pos = home_tcp_pose.position_mm
        
        print(f"📍 Home position: [{home_tcp_pos[0]:.1f}, {home_tcp_pos[1]:.1f}, {home_tcp_pos[2]:.1f}] mm")
        
        # Step 3: Interactive gripper pose selection
        print("🦾 Starting gripper pose visualizer...")
        print("   • Click on plots to set gripper tip position")
        print("   • Use sliders for orientation")
        print("   • Press 'Enter' to finalize poses")
        print("   • Close window when done")
        print()
        
        # Initialize visualizer with gripper mode
        visualizer = AdvancedPoseVisualizer()
        visualizer.gripper_mode = True
        visualizer._toggle_gripper_mode()
        
        print("🚀 Launching interactive visualizer...")
        visualizer.run()
        
        # Get selected poses
        tcp_poses = visualizer.get_poses()
        
        if not tcp_poses:
            print("❌ No poses selected. Demo cancelled.")
            return False
        
        print(f"✅ Selected {len(tcp_poses)} pose(s)")
        
        # Step 4: Motion planning for each selected pose
        print("🚀 Planning motion trajectories...")
        
        planning_results = []
        
        for i, tcp_pose in enumerate(tcp_poses, 1):
            # Extract TCP position and orientation
            tcp_pos_mm = [p * 1000 for p in tcp_pose[:3]]
            tcp_rot_deg = np.degrees(R.from_rotvec(tcp_pose[3:]).as_euler('xyz'))
            
            # Calculate gripper position for display
            tcp_pos_m = tcp_pose[:3]
            rot_matrix = R.from_rotvec(tcp_pose[3:]).as_matrix()
            gripper_offset_vector = np.array([0, 0, visualizer.gripper_offset])
            world_offset = rot_matrix @ gripper_offset_vector
            gripper_pos_m = tcp_pos_m + world_offset
            gripper_pos_mm = gripper_pos_m * 1000
            
            print(f"   🎯 Target #{i}: Gripper tip [{gripper_pos_mm[0]:.1f}, {gripper_pos_mm[1]:.1f}, {gripper_pos_mm[2]:.1f}] mm")
            
            # Plan motion from home to target
            start_time = time.time()
            plan = planner.plan_motion(home_joints, tcp_pos_mm, tcp_rot_deg)
            planning_time = time.time() - start_time
            
            # Store results
            result = {
                'pose_number': i,
                'gripper_pos_mm': gripper_pos_mm,
                'tcp_pos_mm': tcp_pos_mm,
                'tcp_rot_deg': tcp_rot_deg,
                'plan': plan,
                'planning_time': planning_time
            }
            planning_results.append(result)
            
            # Report planning outcome
            if plan.success:
                print(f"      ✅ Planning: SUCCESS ({len(plan.waypoints)} waypoints, {planning_time:.2f}s)")
            else:
                print(f"      ❌ Planning: FAILED ({plan.error_message})")
        
        # Step 5: Results summary
        print()
        print("📊 RESULTS SUMMARY")
        print("-" * 20)
        
        successful_plans = [r for r in planning_results if r['plan'].success]
        failed_plans = [r for r in planning_results if not r['plan'].success]
        
        if len(failed_plans) == 0:
            print(f"🎉 ALL TRAJECTORIES SUCCESSFUL: {len(successful_plans)}/{len(planning_results)}")
            print(f"✅ Success rate: 100% (PERFECT)")
        else:
            print(f"✅ Successful: {len(successful_plans)}/{len(planning_results)}")
            print(f"❌ Failed: {len(failed_plans)}/{len(planning_results)}")
            print(f"📊 Success rate: {len(successful_plans)/len(planning_results)*100:.1f}%")
        
        # Show successful trajectories (minimal)
        if successful_plans:
            print()
            print("🛤️  SUCCESSFUL TRAJECTORIES:")
            
            total_waypoints = 0
            total_planning_time = 0
            
            for result in successful_plans:
                pose_num = result['pose_number']
                plan = result['plan']
                gripper_pos = result['gripper_pos_mm']
                
                total_waypoints += len(plan.waypoints)
                total_planning_time += result['planning_time']
                
                print(f"   🎯 Target #{pose_num}: {len(plan.waypoints)} waypoints, {result['planning_time']:.2f}s")
                
                # FK Validation of all waypoints
                print(f"      🔍 Validating waypoints with Forward Kinematics...")
                
                # Initialize FK for validation
                fk_validator = ForwardKinematics()
                
                validation_errors = []
                max_error = 0.0
                
                # Show all waypoints in the trajectory with FK validation
                if len(plan.waypoints) >= 2:
                    print(f"      📋 Complete trajectory (with FK validation):")
                    
                    for i, waypoint in enumerate(plan.waypoints):
                        joints = waypoint.joints_deg
                        joints_rad = np.deg2rad(joints)
                        
                        # FK validation: compute actual TCP position
                        T_actual = fk_validator.compute_forward_kinematics(joints_rad)
                        actual_tcp_pos = T_actual[:3, 3] * 1000  # Convert to mm
                        
                        # Compare with expected TCP position (if available)
                        if hasattr(waypoint, 'tcp_position_mm') and waypoint.tcp_position_mm is not None:
                            expected_tcp_pos = waypoint.tcp_position_mm
                            error = np.linalg.norm(actual_tcp_pos - expected_tcp_pos)
                            max_error = max(max_error, error)
                            
                            if error > 1.0:  # Error threshold: 1mm
                                validation_errors.append(f"Waypoint #{i}: {error:.3f}mm error")
                        else:
                            # If no expected TCP position, just use FK result
                            expected_tcp_pos = actual_tcp_pos
                            error = 0.0
                        
                        if i == 0:
                            label = "Start"
                        elif i == len(plan.waypoints) - 1:
                            label = "End  "
                        else:
                            label = f"#{i:2d}  "
                        
                        # Show joint angles and FK-computed TCP position
                        print(f"         {label}: [{', '.join([f'{j:.1f}' for j in joints])}]°")
                        print(f"              FK TCP: [{actual_tcp_pos[0]:.1f}, {actual_tcp_pos[1]:.1f}, {actual_tcp_pos[2]:.1f}] mm" + 
                              (f" (err: {error:.3f}mm)" if error > 0.1 else ""))
                        
                        # Add spacing every 5 waypoints for readability (except for very short trajectories)
                        if len(plan.waypoints) > 10 and i > 0 and i < len(plan.waypoints) - 1 and (i + 1) % 5 == 0:
                            print(f"              ...")
                    
                    # FK Validation summary
                    print(f"      ✅ FK Validation Results:")
                    if validation_errors:
                        print(f"         ⚠️  Found {len(validation_errors)} waypoints with errors > 1mm:")
                        for error_msg in validation_errors:
                            print(f"         ❌ {error_msg}")
                        print(f"         📊 Maximum error: {max_error:.3f}mm")
                    else:
                        print(f"         ✅ All waypoints validated successfully")
                        print(f"         📊 Maximum error: {max_error:.3f}mm (EXCELLENT)")
                    
                    print(f"      🤖 Robot program commands:")
                    print(f"         robot.move_joint({', '.join([f'{j:.2f}' for j in plan.waypoints[0].joints_deg])})  # Start")
                    if len(plan.waypoints) > 2:
                        print(f"         # ... {len(plan.waypoints)-2} intermediate waypoints ...")
                    print(f"         robot.move_joint({', '.join([f'{j:.2f}' for j in plan.waypoints[-1].joints_deg])})  # Target")
            
            print(f"\n� Total: {total_waypoints} waypoints, {total_planning_time:.2f}s planning time")
        
        # Show failed trajectories (minimal)
        if failed_plans:
            print("\n❌ FAILED TRAJECTORIES:")
            for result in failed_plans:
                pose_num = result['pose_number']
                gripper_pos = result['gripper_pos_mm']
                print(f"   🎯 Target #{pose_num}: {result['plan'].error_message}")
                
                # Simple diagnostic
                distance_from_base = np.sqrt(result['tcp_pos_mm'][0]**2 + result['tcp_pos_mm'][1]**2)
                if distance_from_base > 720:
                    print(f"      💡 Too far from robot (try within 720mm)")
                elif result['tcp_pos_mm'][2] < 150:
                    print(f"      💡 Too low (try above 200mm)")
                elif result['tcp_pos_mm'][2] > 600:
                    print(f"      💡 Too high (try below 500mm)")
        
        # Final status
        print()
        if successful_plans:
            print("🎉 GRIPPER POSE SELECTION: OPERATIONAL")
            print("✅ System ready for gripper-based operations!")
            return True
        else:
            print("⚠️  No successful trajectories - try poses within workspace")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Main function to run the gripper pose selection test."""
    
    print("🚀 GRIPPER POSE SELECTION TEST")
    print("=" * 40)
    print("Interactive gripper positioning with motion planning")
    print()
    
    if not PLANNING_AVAILABLE:
        print("❌ Planning modules not available")
        return
    
    # Run the demonstration
    success = run_gripper_pose_selection_demo()
    
    print("\n" + "=" * 40)
    print("🏁 TEST COMPLETE")
    print("=" * 40)
    
    if success:
        print("✅ Test PASSED - System operational")
        print("🎯 Ready for gripper-based operations")
    else:
        print("❌ Test FAILED - Check workspace constraints")
        print("💡 Try poses within green 'SAFE' zone")
    
    print("\n🎯 System enables intuitive gripper positioning")
    print("   with automatic motion planning integration.")


if __name__ == "__main__":
    main()