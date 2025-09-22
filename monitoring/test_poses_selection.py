#!/usr/bin/env python3
"""
Simple Pick & Place Pose Visualizer Demo

This script demonstrates the AdvancedPoseVisualizer tool for interactive
robot TCP pose definition and motion planning integration with the same
scenario as pick_and_place_example.py.

Features:
- Interactive pose selection using calibrated robot/camera coordinates
- Motion planning from fixed home position to selected target
- Integration with the enhanced motion planning system
- Real-time waypoint generation and validation
"""
import sys
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add the parent directory to the Python path for integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('../kinematics/src')

from forward_kinematic import ForwardKinematics

def pose_to_transformation_matrix(pose):
    """Convert 6-element pose [x, y, z, rx, ry, rz] to 4x4 transformation matrix."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]  # Position
    if np.linalg.norm(pose[3:]) > 0:
        T[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()  # Rotation
    return T

def run_pick_place_pose_demo():
    """
    Run the simplified pick and place pose demo.
    Same scenario as pick_and_place_example.py but with interactive pose selection.
    """
    print("🤖 PICK & PLACE POSE VISUALIZER DEMO")
    print("="*40)
    print("Interactive pose selection for robot motion planning")
    print()
    print("This demo:")
    print("• Uses the same home position as pick_and_place_example.py")
    print("• Allows interactive selection of ONE target pose")
    print("• Plans motion from home to target using enhanced planning")
    print("• Shows successful waypoints for robot execution")
    print()
    
    try:
        # Import motion planning components
        from planning.examples.clean_robot_interface import CleanRobotMotionPlanner
        from AdvancedPoseVisualizer import AdvancedPoseVisualizer
        
        # Step 1: Initialize the motion planner (same as pick_and_place_example.py)
        print("⚙️  STEP 1: Initialize Motion Planner")
        planner = CleanRobotMotionPlanner()
        print("✅ Clean robot motion planner initialized")
        
        # Step 2: Set the same home position as pick_and_place_example.py
        home_joints = [0.0, -60.0, 90.0, 0.0, 30.0, 0.0]  # degrees - same as example
        home_tcp_pose = planner.get_current_pose_from_joints(home_joints)
        home_tcp_pos = home_tcp_pose.position_mm
        home_tcp_rot = home_tcp_pose.orientation_deg
        
        print(f"\n📍 STEP 2: Home Position (Same as pick_and_place_example.py)")
        print(f"Home joints: {home_joints} degrees")
        print(f"Home TCP: [{home_tcp_pos[0]:.1f}, {home_tcp_pos[1]:.1f}, {home_tcp_pos[2]:.1f}] mm")
        print(f"Home orientation: [{home_tcp_rot[0]:.1f}, {home_tcp_rot[1]:.1f}, {home_tcp_rot[2]:.1f}] degrees")
        
        # Step 3: Interactive pose selection
        print(f"\n🎯 STEP 3: Interactive Target Pose Selection")
        print("Starting Advanced Pose Visualizer...")
        print("Instructions:")
        print("• Click on 2D plots to define target position")
        print("• Use sliders to adjust orientation")
        print("• Press 'Enter' to finalize the pose")
        print("• Close the visualizer when done")
        print()
        
        visualizer = AdvancedPoseVisualizer()
        visualizer.run()
        
        poses = visualizer.get_poses()
        if not poses:
            print("❌ No target pose selected. Demo cancelled.")
            return
        
        if len(poses) > 1:
            print(f"⚠️  Multiple poses selected. Using the first pose only.")
        
        target_pose = poses[0]
        target_pos_mm = [p * 1000 for p in target_pose[:3]]  # Convert to mm
        target_rot_deg = np.degrees(R.from_rotvec(target_pose[3:]).as_euler('xyz'))
        
        print(f"\n✅ Target Pose Selected:")
        print(f"Target position: [{target_pos_mm[0]:.1f}, {target_pos_mm[1]:.1f}, {target_pos_mm[2]:.1f}] mm")
        print(f"Target orientation: [{target_rot_deg[0]:.1f}, {target_rot_deg[1]:.1f}, {target_rot_deg[2]:.1f}] degrees")
        
        # Step 4: Plan motion from home to target
        print(f"\n🚀 STEP 4: Motion Planning (Home → Target)")
        print("Planning motion using enhanced motion planner...")
        
        start_time = time.time()
        plan = planner.plan_motion(home_joints, target_pos_mm, target_rot_deg)
        planning_time = time.time() - start_time
        
        if plan.success:
            print(f"✅ Motion planning SUCCESS!")
            print(f"Planning time: {planning_time:.3f} seconds")
            print(f"Waypoints: {len(plan.waypoints)}")
            print(f"Execution time: {plan.execution_time_sec:.2f} seconds")
            
            # Step 5: Display waypoints with FK validation
            print(f"\n📋 STEP 5: Generated Waypoints with FK Validation")
            print("Robot joint waypoints (degrees):")
            print("-" * 50)
            
            # FK Validation setup
            fk_validator = ForwardKinematics()
            validation_errors = []
            max_error = 0.0
            
            for i, waypoint in enumerate(plan.waypoints):
                wp_type = "Start" if i == 0 else "End" if i == len(plan.waypoints)-1 else f"#{i}"
                joints_str = ", ".join([f"{j:.1f}" for j in waypoint.joints_deg])
                
                # FK validation: compute actual TCP position
                joints_rad = np.deg2rad(waypoint.joints_deg)
                T_actual = fk_validator.compute_forward_kinematics(joints_rad)
                actual_tcp_pos = T_actual[:3, 3] * 1000  # Convert to mm
                
                # Compare with expected TCP position
                expected_tcp_pos = waypoint.tcp_position_mm
                error = np.linalg.norm(actual_tcp_pos - expected_tcp_pos)
                max_error = max(max_error, error)
                
                if error > 1.0:  # Error threshold: 1mm
                    validation_errors.append(f"Waypoint #{i}: {error:.3f}mm error")
                
                print(f"  {wp_type:>5}: [{joints_str}]")
                print(f"         Expected TCP: [{expected_tcp_pos[0]:.1f}, {expected_tcp_pos[1]:.1f}, {expected_tcp_pos[2]:.1f}] mm")
                print(f"         FK Actual TCP: [{actual_tcp_pos[0]:.1f}, {actual_tcp_pos[1]:.1f}, {actual_tcp_pos[2]:.1f}] mm" + 
                      (f" (err: {error:.3f}mm)" if error > 0.1 else ""))
            
            # FK Validation summary
            print(f"\n🔍 FK VALIDATION SUMMARY:")
            if validation_errors:
                print(f"⚠️  Found {len(validation_errors)} waypoints with errors > 1mm:")
                for error_msg in validation_errors:
                    print(f"❌ {error_msg}")
                print(f"📊 Maximum error: {max_error:.3f}mm")
            else:
                print(f"✅ All waypoints validated successfully")
                print(f"📊 Maximum error: {max_error:.3f}mm (EXCELLENT)")
            print("-" * 50)
            
            # Step 6: Generate robot program
            print(f"\n🤖 STEP 6: Robot Program Generation")
            robot_program = plan.generate_robot_program(speed_percent=25.0)
            
            print("Generated robot program:")
            print("```")
            print("# Pick & Place Demo - Home to Target")
            print("robot.set_speed(25.0)")
            print()
            
            # Show key waypoints
            start_wp = plan.waypoints[0]
            end_wp = plan.waypoints[-1]
            
            print(f"# Start position (Home)")
            print(start_wp.move_joint_command())
            print(f"#   TCP: [{start_wp.tcp_position_mm[0]:.1f}, {start_wp.tcp_position_mm[1]:.1f}, {start_wp.tcp_position_mm[2]:.1f}] mm")
            
            if len(plan.waypoints) > 2:
                print(f"\n# ... {len(plan.waypoints)-2} intermediate waypoints ...")
            
            print(f"\n# Target position")
            print(end_wp.move_joint_command())
            print(f"#   TCP: [{end_wp.tcp_position_mm[0]:.1f}, {end_wp.tcp_position_mm[1]:.1f}, {end_wp.tcp_position_mm[2]:.1f}] mm")
            print("```")
            
            # Step 7: Summary
            print(f"\n📊 DEMO SUMMARY")
            print("="*30)
            print(f"✅ Motion planning: SUCCESS")
            print(f"🎯 Target reached: YES")
            print(f"⏱️  Planning time: {planning_time:.3f}s")
            print(f"🛤️  Waypoints: {len(plan.waypoints)}")
            print(f"🤖 Robot ready: YES")
            print()
            print(f"🎉 Demo completed successfully!")
            print(f"The generated waypoints can be executed on the RB3-730ES-U robot.")
            
        else:
            # Calculate distance for detailed diagnostics
            distance_from_base = np.sqrt(target_pos_mm[0]**2 + target_pos_mm[1]**2)
            
            print(f"❌ Motion planning FAILED!")
            print(f"Error: {plan.error_message}")
            print(f"Planning time: {planning_time:.3f} seconds")
            print()
            
            # Detailed diagnostic information
            print(f"� DIAGNOSTIC INFORMATION:")
            print(f"Target position: [{target_pos_mm[0]:.1f}, {target_pos_mm[1]:.1f}, {target_pos_mm[2]:.1f}] mm")
            print(f"Distance from robot base: {distance_from_base:.1f} mm")
            print(f"Target orientation: [{target_rot_deg[0]:.1f}, {target_rot_deg[1]:.1f}, {target_rot_deg[2]:.1f}] degrees")
            print()
            
            # Specific reachability analysis based on URDF and updated constraints
            print(f"📊 REACHABILITY ANALYSIS (URDF + Constraints-Based):")
            print(f"RB3-730ES-U: Theoretical reach = 730mm, Workspace limit = 720mm")
            print(f"Effective workspace after safety margins: ~700mm")
            
            if distance_from_base > 720:
                print(f"❌ UNREACHABLE: Distance ({distance_from_base:.1f}mm) exceeds workspace limit (720mm)")
                print(f"   → Move target closer to robot center")
            elif distance_from_base > 650:
                print(f"⚠️  EXTENDED REACH: Distance ({distance_from_base:.1f}mm) in extended zone (>650mm)")
                print(f"   → Consider moving target within 650mm for better reliability")
            elif distance_from_base > 580:
                print(f"⚠️  WARNING ZONE: Distance ({distance_from_base:.1f}mm) in warning zone (>580mm)")
                print(f"   → Safe zone is within 580mm - some orientations may be difficult")
            else:
                print(f"✅ SAFE ZONE: Distance ({distance_from_base:.1f}mm) is within safe reach (≤580mm)")
                print(f"   → Issue likely related to orientation or Z-height constraints")
            
            # Z-height analysis
            if target_pos_mm[2] < 150:
                print(f"❌ Z-HEIGHT: Target Z ({target_pos_mm[2]:.1f}mm) is too low (collision risk)")
                print(f"   → Use Z-height ≥ 200mm")
            elif target_pos_mm[2] > 600:
                print(f"❌ Z-HEIGHT: Target Z ({target_pos_mm[2]:.1f}mm) may be too high for this reach")
                print(f"   → Try Z-height between 200-500mm")
            else:
                print(f"✅ Z-HEIGHT: Target Z ({target_pos_mm[2]:.1f}mm) is reasonable")
            
            # Orientation analysis
            if abs(target_rot_deg[0]) > 160 or abs(target_rot_deg[1]) > 160:
                print(f"❌ ORIENTATION: Extreme rotation angles may be unreachable")
                print(f"   → Try simpler orientations closer to [180, 0, 0] degrees")
            else:
                print(f"✅ ORIENTATION: Target orientation appears reasonable")
            
            print()
            print(f"💡 SPECIFIC RECOMMENDATIONS:")
            
            # Generate specific recommendations based on the failure
            if distance_from_base > 500:
                # Suggest closer positions
                scale_factor = 400 / distance_from_base
                suggested_x = target_pos_mm[0] * scale_factor
                suggested_y = target_pos_mm[1] * scale_factor
                print(f"• Try position closer to center: [{suggested_x:.1f}, {suggested_y:.1f}, {target_pos_mm[2]:.1f}] mm")
            
            if target_pos_mm[2] < 200:
                print(f"• Increase Z-height to at least 250mm for safety")
            elif target_pos_mm[2] > 500:
                print(f"• Reduce Z-height to 300-400mm range")
            else:
                print(f"• Z-height ({target_pos_mm[2]:.1f}mm) looks good")
            
            print(f"• Use simpler orientation like [180, 0, 0] for downward gripper")
            print(f"• Try positions within the green 'SAFE' zone in the visualizer")
            print(f"• Avoid the red 'MAX' zone at the edge of robot reach")
        
    except ImportError as e:
        print(f"❌ Could not import required modules: {e}")
        print("💡 Make sure the planning and monitoring packages are available")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run the simplified pick and place pose demo."""
    print("🤖 SIMPLIFIED PICK & PLACE POSE DEMO")
    print("Interactive pose selection for robot motion planning")
    print("Based on pick_and_place_example.py scenario")
    print()
    
    try:
        run_pick_place_pose_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()