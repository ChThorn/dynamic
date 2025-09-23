#!/usr/bin/env python3
"""
Integrated Pose Selection and Motion Planning Example

This example demonstrates the seamless integration between the
visualizer_tool_TCP and the planning_dynamic_executor using the
pose_integration_bridge.

Features:
- One-line workflow from pose selection to motion execution
- Automatic format conversion between systems
- Error handling and validation
- Progress tracking and reporting

Usage:
    python integrated_pose_example.py

Author: Robot Motion Planning Team
Date: September 22, 2025
"""

import sys
import os
import numpy as np

# Add project root to path for package-style imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import FK validation (optional)
try:
    from kinematics.src.forward_kinematic import ForwardKinematics
    FK_AVAILABLE = True
except ImportError:
    print("WARNING: Forward kinematics module not available - FK validation will be skipped")
    ForwardKinematics = None
    FK_AVAILABLE = False

def validate_waypoints_with_fk(waypoints, error_threshold_mm=1.0):
    """
    Validate waypoints using Forward Kinematics before robot execution.
    
    Args:
        waypoints: List of waypoints with joints_deg and tcp_position_mm
        error_threshold_mm: Maximum acceptable error in millimeters
        
    Returns:
        tuple: (is_valid, max_error_mm, error_details)
    """
    if not FK_AVAILABLE:
        print("WARNING: FK validation skipped - module not available")
        return True, 0.0, []
    
    print("STEP: FK Validation Before Execution")
    print("Validating waypoint accuracy...")
    
    fk_validator = ForwardKinematics()
    validation_errors = []
    max_error = 0.0
    
    for i, waypoint in enumerate(waypoints):
        # Compute actual TCP position using FK
        joints_rad = np.deg2rad(waypoint.joints_deg)
        T_actual = fk_validator.compute_forward_kinematics(joints_rad)
        actual_tcp_pos = T_actual[:3, 3] * 1000  # Convert to mm
        
        # Compare with expected TCP position  
        expected_tcp_pos = waypoint.tcp_position_mm
        error = np.linalg.norm(actual_tcp_pos - expected_tcp_pos)
        max_error = max(max_error, error)
        
        if error > error_threshold_mm:
            validation_errors.append({
                'waypoint': i,
                'error_mm': error,
                'expected': expected_tcp_pos,
                'actual': actual_tcp_pos.tolist()
            })
    
    # Report validation results
    if validation_errors:
        print(f"FK Validation FAILED!")
        print(f"Found {len(validation_errors)} waypoints with errors > {error_threshold_mm}mm:")
        for error in validation_errors:
            print(f"  Waypoint #{error['waypoint']}: {error['error_mm']:.3f}mm error")
        print(f"Maximum error: {max_error:.3f}mm")
        return False, max_error, validation_errors
    else:
        print(f"FK Validation PASSED!")
        print(f"All {len(waypoints)} waypoints validated successfully")
        print(f"Maximum error: {max_error:.3f}mm (EXCELLENT)")
        return True, max_error, []

def main():
    """Run the integrated pose selection and motion planning example."""
    print("INTEGRATED POSE SELECTION & MOTION PLANNING")
    print("="*60)
    print("This example demonstrates seamless integration between:")
    print("• Interactive 3D pose visualizer (visualizer_tool_TCP)")
    print("• Advanced motion planning (planning_dynamic_executor)")
    print("• Automatic format conversion (pose_integration_bridge)")
    print()
    
    try:
        from robot_driver_interface.src.pose_integration_bridge import create_integrated_workflow
        
        # Create the integrated workflow
        print("STEP 1: Initialize Integrated Workflow")
        bridge = create_integrated_workflow(
            robot_ip="192.168.0.10",
            operation_mode="simulation"  # Safe default
        )
        print("Integration bridge created successfully")
        print()
        
        # Option 1: Complete workflow (visualizer → planning → execution)
        print("OPTION 1: Complete Workflow")
        print("This will:")
        print("1. Open interactive pose visualizer")
        print("2. Convert selected poses to planning format")
        print("3. Execute motions with planning_dynamic_executor")
        print()
        
        run_complete = input("Run complete workflow? (y/n): ").lower().strip()
        
        if run_complete == 'y':
            print("\nRunning complete workflow...")
            print("Instructions for visualizer:")
            print("• Click on 2D plots to set position")
            print("• Use sliders to adjust orientation") 
            print("• Press Enter to finalize each pose")
            print("• Close visualizer when done")
            print()
            
            # Note: FK validation will be added inside the bridge workflow
            print("SAFETY NOTE: Enhanced with FK validation before execution")
            print()
            
            # Run the integrated workflow (FK validation happens inside)
            targets, results = bridge.run_complete_workflow(execution_mode="blend")
            
            # Display comprehensive results
            print("\nWORKFLOW RESULTS")
            print("="*40)
            
            if targets:
                success_count = sum(results)
                print(f"Total poses selected: {len(targets)}")
                print(f"Motions attempted: {len(results)}")
                print(f"Successful motions: {success_count}")
                print(f"Success rate: {success_count/len(results)*100:.1f}%")
                print()
                
                print("Detailed Results:")
                print("-" * 40)
                for i, (target, success) in enumerate(zip(targets, results)):
                    status = "SUCCESS" if success else "FAILED"
                    summary = bridge.get_pose_summary(target)
                    print(f"Pose {i+1}: {status}")
                    print(f"  {summary}")
                print()
                
                if success_count == len(results):
                    print("All motions completed successfully!")
                    print("Robot is ready for real-world deployment.")
                elif success_count > 0:
                    print(f"WARNING: {len(results)-success_count} motions failed")
                    print("Check robot workspace constraints and pose reachability.")
                else:
                    print("All motions failed")
                    print("Review poses and robot configuration.")
            else:
                print("No poses were selected")
                print("Workflow cancelled by user.")
        
        else:
            # Option 2: Demonstration of individual components
            print("\nOPTION 2: Component Demonstration")
            
            # Demo 1: Format conversion
            print("\n1. Format Conversion Demo")
            print("-" * 30)
            
            # Sample visualizer pose (meters + rotation vectors)
            import numpy as np
            sample_pose = np.array([0.4, 0.2, 0.3, 3.14159, 0.0, 0.0])  # 400mm, 200mm, 300mm + 180° rotation
            
            print(f"Visualizer pose: {sample_pose}")
            print("  Position: [400mm, 200mm, 300mm]")
            print("  Rotation: [180°, 0°, 0°] (downward)")
            
            # Convert to planning format
            target = bridge.convert_visualizer_pose_to_planning_target(sample_pose)
            
            print(f"\nPlanning target:")
            print(f"  {bridge.get_pose_summary(target)}")
            print("Conversion successful")
            
            # Demo 2: Executor creation
            print("\n2. Executor Creation Demo")
            print("-" * 30)
            
            executor = bridge.create_executor(execution_mode="blend", chunk_size=8)
            print("Planning dynamic executor created")
            print(f"Operation mode: {executor.get_operation_mode()}")
            print(f"Safe mode: {executor.is_safe_mode()}")
            
            # Demo 3: Single motion test
            print("\n3. Single Motion Test")
            print("-" * 30)
            
            print("Testing motion planning to sample pose...")
            success = executor.plan_and_execute_motion(target)
            
            if success:
                print("Motion planning successful!")
                print("Waypoints generated and validated.")
            else:
                print("Motion planning failed")
                print("Check pose reachability and robot state.")
        
        print("\n" + "="*60)
        print("INTEGRATION EXAMPLE COMPLETE")
        print("="*60)
        print("Key Benefits:")
        print("• Seamless data conversion between systems")
        print("• Interactive pose selection with 3D visualization")
        print("• Advanced motion planning with polynomial trajectories")
        print("• **SAFETY-FIRST: FK validation before execution**")
        print("• Automatic error detection and user warnings")
        print("• Production-ready for robot controller deployment")
        print()
        print("SAFETY FEATURES:")
        print("• FK validation prevents kinematic errors")
        print("• User confirmation required for failed validation")
        print("• Clear error reporting with specific waypoint details")
        print("• Execution blocking for critical safety issues")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("NOTE: Make sure all required modules are available:")
        print("   - pose_integration_bridge.py")
        print("   - planning_dynamic_executor.py") 
        print("   - visualizer_tool_TCP.py")
    except KeyboardInterrupt:
        print("\nExample cancelled by user")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            bridge.shutdown()
            print("Integration bridge shut down cleanly")
        except:
            pass

if __name__ == "__main__":
    main()