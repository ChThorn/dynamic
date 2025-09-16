#!/usr/bin/env python3
"""
Monitoring Package Demo

This script demonstrates the AdvancedPoseVisualizer tool for interactive
robot TCP pose definition and capture.
"""

import sys
import os

# Add the parent directory to the Python path for future integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def show_menu():
    """Display the demo menu."""
    print("🚀 MONITORING PACKAGE DEMO")
    print("="*30)
    print()
    print("Choose an option:")
    print("1. 🤖 Pick & Place Pose Visualizer")
    print("2. 🚀 Live Motion Planning Integration")
    print("3. ✅ Predefined Success Demo")
    print("4. ⚖️  IK Method Comparison (Basic vs C-space)")
    print("5. 📖 Show Integration Guide")
    print("6. ❌ Exit")
    print()

def run_pose_visualizer():
    """Run the interactive pose visualizer."""
    print("\n🤖 PICK & PLACE POSE VISUALIZER")
    print("="*35)
    print("Starting Pick & Place Pose Visualizer...")
    print("\nThis tool allows you to:")
    print("• Define pick & place poses with workspace constraints")
    print("• Default downward gripper orientation for reliable grasping")
    print("• Automatic workspace validation for motion planning success")
    print("• Export poses optimized for robot execution")
    print()
    
    try:
        from AdvancedPoseVisualizer import AdvancedPoseVisualizer
        
        visualizer = AdvancedPoseVisualizer()
        visualizer.run()
        
        poses = visualizer.get_poses()
        if poses:
            print(f"\n✅ Captured {len(poses)} pick & place poses:")
            for i, pose in enumerate(poses, 1):
                pos_mm = [p * 1000 for p in pose[:3]]
                print(f"   Pose {i}: [{pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f}] mm (downward grip)")
            print(f"\n💾 Poses exported to 'pick_place_poses.json'")
            print(f"🤖 These poses are ready for motion planning and robot execution")
        else:
            print("\n⚠️  No poses captured.")
            
    except Exception as e:
        print(f"❌ Error running visualizer: {e}")

def run_live_integration():
    """Run live integration between pose visualizer and motion planning."""
    print("\n🤖 LIVE MOTION PLANNING INTEGRATION")
    print("="*40)
    print("This demo shows the complete workflow:")
    print("1. Define poses interactively")
    print("2. Plan motions automatically")
    print("3. Generate robot programs")
    print()
    print("⚠️  WORKSPACE GUIDELINES:")
    print("   • Use full robot workspace (-0.7 to +0.7m)")
    print("   • Avoid extreme joint angles")
    print("   • Prefer poses closer to robot center for better success")
    print("   • Simple orientations work better than complex rotations")
    print()
    
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        
        # Import motion planning components
        from planning.src.motion_planner import MotionPlanner, PlanningStatus
        from kinematics.src.forward_kinematic import ForwardKinematics
        from kinematics.src.inverse_kinematic import InverseKinematics
        
        def pose_to_matrix(pose):
            """Convert 6-element pose [x, y, z, rx, ry, rz] to 4x4 transformation matrix."""
            T = np.eye(4)
            T[:3, 3] = pose[:3]  # Position
            if np.linalg.norm(pose[3:]) > 0:
                T[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()  # Rotation
            return T
        
        # Step 1: Capture poses interactively
        print("🎯 STEP 1: INTERACTIVE POSE CAPTURE")
        print("Define poses using the visualizer...")
        
        from AdvancedPoseVisualizer import AdvancedPoseVisualizer
        visualizer = AdvancedPoseVisualizer()
        visualizer.run()
        poses = visualizer.get_poses()
        
        if not poses:
            print("❌ No poses captured. Integration demo cancelled.")
            return
        
        print(f"✅ Captured {len(poses)} poses")
        for i, pose in enumerate(poses, 1):
            pos_mm = [p * 1000 for p in pose[:3]]
            print(f"   Pose {i}: [{pos_mm[0]:.1f}, {pos_mm[1]:.1f}, {pos_mm[2]:.1f}] mm")
        
        # Initialize motion planner (using basic IK for better reliability)
        print(f"\n⚙️  STEP 2: MOTION PLANNER INITIALIZATION")
        fk = ForwardKinematics()
        ik = InverseKinematics(fk)
        planner = MotionPlanner(fk, ik)
        # Note: Not enabling C-space analysis for better pose compatibility
        print("✅ Motion planner ready (using reliable basic DLS IK)")
        
        # Step 3: Plan motions using proper robot procedure
        print(f"\n🚀 STEP 3: TRAJECTORY PLANNING (Proper Robot Procedure)")
        
        if len(poses) == 1:
            print("🎯 Single pose motion: Planning from home position to target")
            print("Following real robot motion planning procedure:")
            print("• Using fixed home position as starting point")
            print("• Planning direct motion to target pose")
            print("• Validating with forward kinematics")
        else:
            print("� Multi-pose motion: Planning sequential trajectory")
            print("Following real robot motion planning procedure:")
            print("• Using sequential IK solving with motion continuity")
            print("• Each pose uses previous solution as initial guess")
            print("• Validating with forward kinematics")
        
        # Convert poses to transformation matrices
        pose_matrices = []
        for pose in poses:
            T = pose_to_matrix(pose)
            pose_matrices.append(T)
        
        # Motion planning based on number of poses
        successful_motions = []
        failed_motions = []
        
        try:
            if len(poses) == 1:
                # Single pose: Plan from home position to target
                print(f"\n🎯 Single Pose Motion Planning (Home → Target)")
                target_pose = pose_matrices[0]
                
                # Define fixed home position (not random)
                home_joints = np.array([0, -np.pi/2, np.pi/2, 0, np.pi/2, 0])
                home_pose = planner.fk.compute_forward_kinematics(home_joints)
                
                print(f"   🏠 Starting from fixed home position")
                print(f"   🎯 Planning motion to target pose")
                
                # Plan motion from home pose to target pose
                result = planner.plan_cartesian_motion(home_pose, target_pose, current_joints=home_joints)
                
                if result.status == PlanningStatus.SUCCESS:
                    print(f"   ✅ Single pose motion SUCCESS!")
                    print(f"   Total waypoints: {len(result.plan.joint_waypoints)}")
                    print(f"   Planning time: {result.planning_time:.3f}s")
                    print(f"   Strategy: Home position → Target pose")
                    successful_motions = [(0, result)]
                else:
                    print(f"   ❌ Single pose motion failed: {result.error_message}")
                    failed_motions = [(0, result.error_message)]
                    
            else:
                # Multiple poses: Use sequential planning
                print(f"\n🤖 Multi-Pose Sequential Motion Planning")
                result = planner.plan_sequential_cartesian_motion(pose_matrices)
                
                if result.status == PlanningStatus.SUCCESS:
                    print(f"   ✅ Sequential motion planning SUCCESS!")
                    print(f"   Total waypoints: {len(result.plan.joint_waypoints)}")
                    print(f"   Planning time: {result.planning_time:.3f}s")
                    print(f"   Strategy: Sequential IK with motion continuity")
                    successful_motions = [(0, result)]
                else:
                    print(f"   ❌ Sequential planning failed: {result.error_message}")
                    
                    # Fallback: Use pairwise planning
                    print(f"\n🔄 Fallback: Pairwise Motion Planning")
                    
                    for i in range(len(poses) - 1):
                        start_pose = poses[i]
                        goal_pose = poses[i + 1]
                        
                        print(f"\n🎯 Planning Motion {i+1}: Pose {i+1} → Pose {i+2}")
                        
                        try:
                            # Convert poses to transformation matrices
                            T_start = pose_to_matrix(start_pose)
                            T_goal = pose_to_matrix(goal_pose)
                            
                            # Use proper robot procedure with continuity
                            if i == 0:
                                # First motion: start from fixed home position
                                home_joints = np.array([0, -np.pi/2, np.pi/2, 0, np.pi/2, 0])
                                result = planner.plan_cartesian_motion(T_start, T_goal, current_joints=home_joints)
                            else:
                                # Subsequent motions: use previous result's end joints
                                if successful_motions:
                                    prev_result = successful_motions[-1][1]
                                    if prev_result.plan and prev_result.plan.joint_waypoints:
                                        last_joints = prev_result.plan.joint_waypoints[-1]
                                        result = planner.plan_cartesian_motion(T_start, T_goal, current_joints=last_joints)
                                    else:
                                        home_joints = np.array([0, -np.pi/2, np.pi/2, 0, np.pi/2, 0])
                                        result = planner.plan_cartesian_motion(T_start, T_goal, current_joints=home_joints)
                                else:
                                    home_joints = np.array([0, -np.pi/2, np.pi/2, 0, np.pi/2, 0])
                                    result = planner.plan_cartesian_motion(T_start, T_goal, current_joints=home_joints)
                            
                            if result.status == PlanningStatus.SUCCESS:
                                successful_motions.append((i, result))
                                print(f"   ✅ Success: {len(result.plan.joint_waypoints)} waypoints")
                                print(f"   Strategy: {result.plan.strategy_used}")
                                print(f"   Planning time: {result.planning_time:.3f}s")
                            else:
                                failed_motions.append((i, result.error_message))
                                print(f"   ❌ Failed: {result.error_message}")
                                
                        except Exception as e:
                            failed_motions.append((i, str(e)))
                            print(f"   ❌ Exception: {e}")
        
        except Exception as e:
            print(f"❌ Motion planning error: {e}")
            successful_motions = []
            failed_motions = [(0, str(e))]
        
        # Step 4: Generate robot program
        print(f"\n🤖 STEP 4: ROBOT PROGRAM GENERATION")
        if successful_motions:
            print("Generated robot execution program:")
            print("```")
            print("# RB3 Robot Program - Interactive Poses")
            print("robot.gripper_open()")
            
            if len(poses) == 1:
                # Single pose: Home → Target
                result = successful_motions[0][1]
                print(f"\n# Single Motion: Home Position → Target Pose")
                
                waypoints = result.plan.joint_waypoints
                joint_deg = np.degrees(waypoints[0])
                print(f"robot.move_joint({joint_deg[0]:.1f}, {joint_deg[1]:.1f}, "
                      f"{joint_deg[2]:.1f}, {joint_deg[3]:.1f}, {joint_deg[4]:.1f}, {joint_deg[5]:.1f})  # Start (Home)")
                
                if len(waypoints) > 2:
                    print(f"# ... {len(waypoints)-2} intermediate waypoints ...")
                
                joint_deg = np.degrees(waypoints[-1])
                print(f"robot.move_joint({joint_deg[0]:.1f}, {joint_deg[1]:.1f}, "
                      f"{joint_deg[2]:.1f}, {joint_deg[3]:.1f}, {joint_deg[4]:.1f}, {joint_deg[5]:.1f})  # Target")
            else:
                # Multiple poses: Sequential motions
                for motion_idx, result in successful_motions:
                    if len(successful_motions) == 1:
                        # Sequential planning result
                        print(f"\n# Sequential Motion: All poses")
                    else:
                        # Pairwise planning results
                        print(f"\n# Motion {motion_idx+1}: Pose {motion_idx+1} → Pose {motion_idx+2}")
                    
                    waypoints = result.plan.joint_waypoints
                    joint_deg = np.degrees(waypoints[0])
                    print(f"robot.move_joint({joint_deg[0]:.1f}, {joint_deg[1]:.1f}, "
                          f"{joint_deg[2]:.1f}, {joint_deg[3]:.1f}, {joint_deg[4]:.1f}, {joint_deg[5]:.1f})  # Start")
                    
                    if len(waypoints) > 2:
                        print(f"# ... {len(waypoints)-2} intermediate waypoints ...")
                    
                    joint_deg = np.degrees(waypoints[-1])
                    print(f"robot.move_joint({joint_deg[0]:.1f}, {joint_deg[1]:.1f}, "
                          f"{joint_deg[2]:.1f}, {joint_deg[3]:.1f}, {joint_deg[4]:.1f}, {joint_deg[5]:.1f})  # End")
            
            print("```")
        
        # Step 5: Summary
        print(f"\n📊 INTEGRATION SUMMARY")
        if len(poses) == 1:
            total_motions = 1  # Single motion: Home → Target
        else:
            total_motions = len(poses) - 1  # Sequential motions between poses
        success_count = len(successful_motions)
        print(f"   Total poses captured: {len(poses)}")
        print(f"   Motions planned: {total_motions}")
        print(f"   Successful motions: {success_count}")
        print(f"   Success rate: {success_count/total_motions*100:.1f}%")
        
        if failed_motions:
            print(f"\n❌ FAILED MOTIONS:")
            for motion_idx, error in failed_motions:
                print(f"   Motion {motion_idx+1}: {error}")
        
        if success_count > 0:
            print(f"\n🎉 Integration successful! Robot programs ready for execution.")
        else:
            print(f"\n⚠️  No successful motions. Try adjusting poses within robot workspace.")
            print(f"\n💡 TIPS FOR SUCCESS:")
            print(f"   • Use poses closer to robot center")
            print(f"   • Keep Z-height between 200-400mm")
            print(f"   • Avoid positions beyond 500mm from base")
            print(f"   • Try simpler orientations (less rotation)")
            
    except ImportError as e:
        print(f"❌ Could not import motion planning modules: {e}")
        print("💡 Make sure the planning and kinematics packages are available")
    except Exception as e:
        print(f"❌ Error during integration: {e}")

def run_predefined_success_demo():
    """Run integration with known working poses to demonstrate success."""
    print("\n✅ PREDEFINED SUCCESS DEMO")
    print("="*30)
    print("This demo uses known reachable poses to show successful integration.")
    print()
    
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        
        # Import motion planning components
        from planning.src.motion_planner import MotionPlanner, PlanningStatus
        from kinematics.src.forward_kinematic import ForwardKinematics
        from kinematics.src.inverse_kinematic import InverseKinematics
        
        def pose_to_matrix(pose):
            """Convert 6-element pose [x, y, z, rx, ry, rz] to 4x4 transformation matrix."""
            T = np.eye(4)
            T[:3, 3] = pose[:3]  # Position
            if np.linalg.norm(pose[3:]) > 0:
                T[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()  # Rotation
            return T
        
        # Known working poses (within robot workspace)
        working_poses = [
            [0.3, 0.0, 0.3, 0.0, 0.0, 0.0],      # Conservative center pose
            [0.3, 0.0, 0.25, 0.0, 0.0, 0.0],     # Lower position
            [0.25, 0.1, 0.3, 0.0, 0.0, 0.3],     # Small movement with rotation
        ]
        
        print(f"🎯 USING {len(working_poses)} KNOWN WORKING POSES:")
        for i, pose in enumerate(working_poses, 1):
            pos_mm = [p * 1000 for p in pose[:3]]
            rot_deg = [np.degrees(r) for r in pose[3:]]
            print(f"   Pose {i}: [{pos_mm[0]:.0f}, {pos_mm[1]:.0f}, {pos_mm[2]:.0f}] mm, "
                  f"[{rot_deg[0]:.0f}, {rot_deg[1]:.0f}, {rot_deg[2]:.0f}]°")
        
        # Initialize motion planner (using basic IK by default)
        print(f"\n⚙️  MOTION PLANNER INITIALIZATION")
        fk = ForwardKinematics()
        ik = InverseKinematics(fk)
        planner = MotionPlanner(fk, ik)
        print("✅ Motion planner ready (using basic DLS IK)")
        
        # Plan motions between poses
        print(f"\n🚀 TRAJECTORY PLANNING")
        successful_motions = []
        failed_motions = []
        
        for i in range(len(working_poses) - 1):
            start_pose = working_poses[i]
            goal_pose = working_poses[i + 1]
            
            print(f"\n🎯 Planning Motion {i+1}: Pose {i+1} → Pose {i+2}")
            
            try:
                # Convert poses to transformation matrices
                T_start = pose_to_matrix(start_pose)
                T_goal = pose_to_matrix(goal_pose)
                
                # Plan motion
                result = planner.plan_cartesian_motion(T_start, T_goal)
                
                if result.status == PlanningStatus.SUCCESS:
                    successful_motions.append((i, result))
                    print(f"   ✅ Success: {len(result.plan.joint_waypoints)} waypoints")
                    print(f"   Strategy: {result.plan.strategy_used}")
                    print(f"   Planning time: {result.planning_time:.3f}s")
                else:
                    failed_motions.append((i, result.error_message))
                    print(f"   ❌ Failed: {result.error_message}")
                    
            except Exception as e:
                failed_motions.append((i, str(e)))
                print(f"   ❌ Exception: {e}")
        
        # Summary
        print(f"\n📊 SUCCESS DEMO SUMMARY")
        total_motions = len(working_poses) - 1
        success_count = len(successful_motions)
        print(f"   Total motions: {total_motions}")
        print(f"   Successful: {success_count}")
        print(f"   Success rate: {success_count/total_motions*100:.1f}%")
        
        if success_count > 0:
            print(f"\n🎉 SUCCESS! This shows the integration DOES work with proper poses!")
        else:
            print(f"\n⚠️  Even conservative poses failed. Check system configuration.")
            
    except Exception as e:
        print(f"❌ Error during success demo: {e}")

def run_ik_comparison_demo():
    """Compare basic IK vs C-space enhanced IK performance."""
    print("\n⚖️  IK METHOD COMPARISON DEMO")
    print("="*35)
    print("This demo compares Basic DLS IK vs C-space Enhanced IK performance.")
    print()
    
    try:
        import numpy as np
        from scipy.spatial.transform import Rotation as R
        import time
        
        # Import motion planning components
        from planning.src.motion_planner import MotionPlanner, PlanningStatus
        from kinematics.src.forward_kinematic import ForwardKinematics
        from kinematics.src.inverse_kinematic import InverseKinematics
        
        def pose_to_matrix(pose):
            """Convert 6-element pose [x, y, z, rx, ry, rz] to 4x4 transformation matrix."""
            T = np.eye(4)
            T[:3, 3] = pose[:3]  # Position
            if np.linalg.norm(pose[3:]) > 0:
                T[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()  # Rotation
            return T
        
        # Test poses (within robot workspace)
        test_poses = [
            [0.3, 0.0, 0.3, 0.0, 0.0, 0.0],      # Conservative center pose
            [0.3, 0.0, 0.25, 0.0, 0.0, 0.0],     # Lower position
            [0.25, 0.1, 0.3, 0.0, 0.0, 0.3],     # Small movement with rotation
            [0.35, -0.1, 0.35, 0.0, 0.2, 0.0],   # Extended pose
        ]
        
        print(f"🎯 TESTING WITH {len(test_poses)} POSES:")
        for i, pose in enumerate(test_poses, 1):
            pos_mm = [p * 1000 for p in pose[:3]]
            rot_deg = [np.degrees(r) for r in pose[3:]]
            print(f"   Pose {i}: [{pos_mm[0]:.0f}, {pos_mm[1]:.0f}, {pos_mm[2]:.0f}] mm, "
                  f"[{rot_deg[0]:.0f}, {rot_deg[1]:.0f}, {rot_deg[2]:.0f}]°")
        
        # Initialize motion planner
        print(f"\n⚙️  MOTION PLANNER INITIALIZATION")
        fk = ForwardKinematics()
        ik = InverseKinematics(fk)
        planner = MotionPlanner(fk, ik)
        print("✅ Motion planner ready")
        
        # Test 1: Basic IK Performance
        print(f"\n🔄 TEST 1: BASIC DLS IK (Current Method)")
        print("-" * 40)
        
        basic_times = []
        basic_successes = 0
        
        for i in range(len(test_poses) - 1):
            start_pose = test_poses[i]
            goal_pose = test_poses[i + 1]
            
            print(f"\n🎯 Motion {i+1}: Pose {i+1} → Pose {i+2}")
            
            try:
                T_start = pose_to_matrix(start_pose)
                T_goal = pose_to_matrix(goal_pose)
                
                # Time the motion planning with basic IK
                start_time = time.time()
                result = planner.plan_cartesian_motion(T_start, T_goal)
                planning_time = time.time() - start_time
                
                basic_times.append(planning_time)
                
                if result.status == PlanningStatus.SUCCESS:
                    basic_successes += 1
                    print(f"   ✅ Success: {len(result.plan.joint_waypoints)} waypoints")
                    print(f"   Planning time: {planning_time:.3f}s")
                else:
                    print(f"   ❌ Failed: {result.error_message}")
                    print(f"   Planning time: {planning_time:.3f}s")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                basic_times.append(0)
        
        # Test 2: C-space Enhanced IK Performance
        print(f"\n🚀 TEST 2: C-SPACE ENHANCED IK")
        print("-" * 40)
        print("Enabling configuration space analysis...")
        
        # Enable C-space analysis
        planner.enable_configuration_space_analysis(build_maps=True)
        print("✅ C-space analysis enabled")
        
        cspace_times = []
        cspace_successes = 0
        
        for i in range(len(test_poses) - 1):
            start_pose = test_poses[i]
            goal_pose = test_poses[i + 1]
            
            print(f"\n🎯 Motion {i+1}: Pose {i+1} → Pose {i+2}")
            
            try:
                T_start = pose_to_matrix(start_pose)
                T_goal = pose_to_matrix(goal_pose)
                
                # Time the motion planning with C-space IK
                start_time = time.time()
                result = planner.plan_cartesian_motion(T_start, T_goal)
                planning_time = time.time() - start_time
                
                cspace_times.append(planning_time)
                
                if result.status == PlanningStatus.SUCCESS:
                    cspace_successes += 1
                    print(f"   ✅ Success: {len(result.plan.joint_waypoints)} waypoints")
                    print(f"   Planning time: {planning_time:.3f}s")
                else:
                    print(f"   ❌ Failed: {result.error_message}")
                    print(f"   Planning time: {planning_time:.3f}s")
                    
            except Exception as e:
                print(f"   ❌ Exception: {e}")
                cspace_times.append(0)
        
        # Performance Comparison
        print(f"\n📊 PERFORMANCE COMPARISON")
        print("=" * 40)
        total_motions = len(test_poses) - 1
        
        basic_avg_time = sum(basic_times) / len(basic_times) if basic_times else 0
        cspace_avg_time = sum(cspace_times) / len(cspace_times) if cspace_times else 0
        speedup = basic_avg_time / cspace_avg_time if cspace_avg_time > 0 else 0
        
        print(f"📈 BASIC DLS IK:")
        print(f"   Success rate: {basic_successes}/{total_motions} ({basic_successes/total_motions*100:.1f}%)")
        print(f"   Average time: {basic_avg_time:.3f}s")
        
        print(f"🚀 C-SPACE ENHANCED IK:")
        print(f"   Success rate: {cspace_successes}/{total_motions} ({cspace_successes/total_motions*100:.1f}%)")
        print(f"   Average time: {cspace_avg_time:.3f}s")
        
        if speedup > 0:
            print(f"⚡ SPEEDUP: {speedup:.2f}x faster with C-space")
        
        if cspace_successes >= basic_successes and cspace_avg_time < basic_avg_time:
            print(f"🎉 C-space IK shows improved performance!")
        elif basic_successes > cspace_successes:
            print(f"⚠️  Basic IK had better success rate")
        else:
            print(f"📊 Results are comparable between methods")
            
    except Exception as e:
        print(f"❌ Error during IK comparison: {e}")

def show_integration_guide():
    """Show guide for integrating with motion planning."""
    print("\n📖 MOTION PLANNING INTEGRATION GUIDE")
    print("="*40)
    print("""
The AdvancedPoseVisualizer generates TCP poses that can be integrated
with the motion planning system. Here's how:

📝 BASIC INTEGRATION:
   1. Run the pose visualizer to define poses
   2. Import the motion planning modules:
      
      from planning.src.motion_planner import MotionPlanner
      from kinematics.src.forward_kinematic import ForwardKinematics  
      from kinematics.src.inverse_kinematic import InverseKinematics
      
   3. Initialize the motion planner:
   
      fk = ForwardKinematics()
      ik = InverseKinematics(fk)
      planner = MotionPlanner(fk, ik)
      
   4. Convert poses and plan motions:
   
      # Convert 6-element pose to 4x4 matrix
      T = pose_to_transformation_matrix(pose)
      
      # Plan motion between poses
      result = planner.plan_cartesian_motion(T_start, T_goal)

🎯 POSE FORMAT:
   • Visualizer output: [x, y, z, rx, ry, rz] (meters, radians)
   • Motion planner input: 4x4 transformation matrices
   
📊 INTEGRATION WORKFLOW:
   1. Define poses interactively
   2. Plan collision-free trajectories  
   3. Generate robot execution programs
   4. Validate reachability and constraints

🔧 HELPER FUNCTIONS NEEDED:
   • pose_to_transformation_matrix() - Convert format
   • Proper error handling for IK failures
   • Workspace boundary validation

For complete examples, see the planning/examples/ directory.
""")

def main():
    """Run the monitoring package demo with menu."""
    print("🤖 ROBOT MOTION PLANNING MONITORING PACKAGE")
    print("Interactive tools for TCP pose definition and motion planning integration")
    print()
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == "1":
                run_pose_visualizer()
            elif choice == "2":
                run_live_integration()
            elif choice == "3":
                run_predefined_success_demo()
            elif choice == "4":
                run_ik_comparison_demo()
            elif choice == "5":
                show_integration_guide()
            elif choice == "6":
                print("\n👋 Goodbye!")
                break
            else:
                print("\n❌ Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
        
        # Wait for user before showing menu again
        if choice in ["1", "2", "3", "4", "5"]:
            input("\nPress Enter to continue...")
            print("\n")

if __name__ == "__main__":
    main()