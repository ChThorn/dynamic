#!/usr/bin/env python3
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
    
    print("STEP: Running FK Validation...")
    
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
        print(f"FK Validation FAILED! Found {len(validation_errors)} waypoints with errors > {error_threshold_mm}mm.")
        for error in validation_errors:
            print(f"  - Waypoint #{error['waypoint']}: {error['error_mm']:.3f}mm error")
        print(f"  Max error: {max_error:.3f}mm")
        return False, max_error, validation_errors
    else:
        print(f"FK Validation PASSED. All {len(waypoints)} waypoints are accurate (max error: {max_error:.3f}mm).")
        return True, max_error, []

def main():
    """Run the integrated pose selection and motion planning example."""
    print("="*60)
    print("INTEGRATED POSE SELECTION & MOTION PLANNING")
    print("="*60)
    print("This example runs a complete workflow: visualizer -> planning -> execution.")
    print()
    
    try:
        from robot_driver_interface.src.pose_integration_bridge import create_integrated_workflow
        
        # Create the integrated workflow
        print("STEP 1: Initializing Integrated Workflow...")
        bridge = create_integrated_workflow(
            robot_ip="192.168.0.10",
            operation_mode="simulation"  # Safe default
        )
        print("...OK")
        print()
        
        # Run the complete workflow by default
        print("STEP 2: Running Complete Workflow...")
        print("--> Opening interactive visualizer. Select poses and close when done.")
        
        # Run the integrated workflow (FK validation happens inside)
        targets, results = bridge.run_complete_workflow(execution_mode="blend")
        
        # Display comprehensive results
        print("\nSTEP 3: WORKFLOW RESULTS")
        print("="*40)
        
        if targets:
            success_count = sum(results)
            print(f"Summary: {success_count}/{len(results)} motions succeeded ({success_count/len(results)*100:.1f}%).")
            print("-" * 40)
            
            for i, (target, success) in enumerate(zip(targets, results)):
                status = "SUCCESS" if success else "FAILED"
                summary = bridge.get_pose_summary(target)
                print(f"Pose {i+1}: {status} | {summary}")
            print()
            
            if success_count == len(results):
                print("All motions completed successfully!")
            else:
                print(f"WARNING: {len(results)-success_count} motions failed. Check reachability.")
        else:
            print("No poses were selected. Workflow cancelled.")
        
        print("\n" + "="*60)
        print("INTEGRATION EXAMPLE COMPLETE")
        print("="*60)
        
    except ImportError as e:
        print(f"Import error: {e}")
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