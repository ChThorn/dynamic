#!/usr/bin/env python3
"""
Pose Integration Bridge

This module provides seamless integration between the visualizer_tool_TCP
and planning_dynamic_executor by handling data format conversions and
providing convenient wrapper functions.

Features:
- Automatic pose format conversion
- Batch pose processing
- Validation and error handling
- Simplified workflow integration

Author: Robot Motion Planning Team
Date: September 22, 2025
"""

import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation as R
from typing import List, Tuple, Optional, Union
import logging

# Add paths for imports - using absolute paths for reliability
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(current_dir, '..', '..')
monitoring_dir = os.path.join(base_dir, 'monitoring')

sys.path.append(current_dir)
sys.path.append(monitoring_dir)

# Import the core modules
from planning_dynamic_executor import PlanningDynamicExecutor, PlanningTarget, create_planning_target
from visualizer_tool_TCP import visualizer_tool_TCP

logger = logging.getLogger("PoseIntegrationBridge")

class PoseIntegrationBridge:
    """
    Integration bridge between visualizer_tool_TCP and planning_dynamic_executor.
    
    This class provides seamless conversion between the pose formats used by
    the visualizer (meters + rotation vectors) and the planning executor 
    (millimeters + degree rotations).
    """
    
    def __init__(self, robot_ip: str = "192.168.0.10", operation_mode: str = "simulation"):
        """
        Initialize the pose integration bridge.
        
        Args:
            robot_ip: IP address of the robot controller
            operation_mode: "simulation" or "real" mode
        """
        self.robot_ip = robot_ip
        self.operation_mode = operation_mode
        self.executor = None
        self.visualizer = None
        
        logger.info(f"Pose integration bridge initialized for robot at {robot_ip}")
    
    def convert_visualizer_pose_to_planning_target(self, visualizer_pose: np.ndarray, 
                                                 gripper_mode: bool = False) -> PlanningTarget:
        """
        Convert a pose from visualizer format to planning target format.
        
        Args:
            visualizer_pose: [x, y, z, rx, ry, rz] in meters and rotation vectors
            gripper_mode: Whether this is a gripper position or TCP position
            
        Returns:
            PlanningTarget object ready for motion planning
        """
        if len(visualizer_pose) != 6:
            raise ValueError(f"Expected 6-element pose, got {len(visualizer_pose)}")
        
        # Convert position from meters to millimeters
        position_mm = [float(visualizer_pose[0] * 1000), 
                      float(visualizer_pose[1] * 1000), 
                      float(visualizer_pose[2] * 1000)]
        
        # Convert rotation vector to Euler angles in degrees
        rotation_vec = visualizer_pose[3:6]
        if np.linalg.norm(rotation_vec) > 0:
            rotation_deg = np.degrees(R.from_rotvec(rotation_vec).as_euler('xyz')).tolist()
        else:
            rotation_deg = [180.0, 0.0, 0.0]  # Default downward orientation
        
        return create_planning_target(
            x_mm=position_mm[0],
            y_mm=position_mm[1], 
            z_mm=position_mm[2],
            rx_deg=rotation_deg[0],
            ry_deg=rotation_deg[1],
            rz_deg=rotation_deg[2],
            gripper_mode=gripper_mode
        )
    
    def convert_planning_target_to_visualizer_pose(self, planning_target) -> np.ndarray:
        """
        Convert planning target back to visualizer pose format (for round-trip validation).
        
        Args:
            planning_target: PlanningTarget from planning_dynamic_executor
            
        Returns:
            np.ndarray: Pose in visualizer format [x,y,z,rx,ry,rz] (meters, rotation vectors)
        """
        try:
            # Extract position (mm to meters)
            pos_m = np.array(planning_target.tcp_position_mm) / 1000.0
            
            # Extract rotation (degrees to rotation vector)
            rot_deg = np.array(planning_target.tcp_rotation_deg)
            rot_rad = np.radians(rot_deg)
            
            # Convert Euler angles to rotation vector
            if np.allclose(rot_rad, 0):
                rot_vec = np.array([0, 0, 0])
            else:
                rotation = R.from_euler('xyz', rot_rad)
                rot_vec = rotation.as_rotvec()
            
            # Combine position and rotation
            pose = np.concatenate([pos_m, rot_vec])
            
            return pose
            
        except Exception as e:
            self.logger.error(f"Failed to convert planning target to visualizer pose: {e}")
            raise ValueError(f"Invalid planning target conversion: {e}")
    
    def convert_multiple_poses(self, visualizer_poses: List[np.ndarray], 
                             gripper_mode: bool = False) -> List[PlanningTarget]:
        """
        Convert multiple poses from visualizer to planning format.
        
        Args:
            visualizer_poses: List of [x, y, z, rx, ry, rz] poses
            gripper_mode: Whether these are gripper positions
            
        Returns:
            List of PlanningTarget objects
        """
        planning_targets = []
        for i, pose in enumerate(visualizer_poses):
            try:
                target = self.convert_visualizer_pose_to_planning_target(pose, gripper_mode)
                planning_targets.append(target)
                logger.debug(f"Converted pose {i+1}: {pose[:3]*1000} mm")
            except Exception as e:
                logger.error(f"Failed to convert pose {i+1}: {e}")
                raise
        
        logger.info(f"Successfully converted {len(planning_targets)} poses")
        return planning_targets
    
    def run_interactive_pose_selection(self) -> List[PlanningTarget]:
        """
        Run the interactive pose visualizer and return planning targets.
        
        Returns:
            List of PlanningTarget objects ready for motion planning
        """
        logger.info("Starting interactive pose selection...")
        
        # Create and run visualizer
        self.visualizer = visualizer_tool_TCP()
        self.visualizer.run()
        
        # Get poses from visualizer
        visualizer_poses = self.visualizer.get_poses()
        
        if not visualizer_poses:
            logger.warning("No poses were selected")
            return []
        
        logger.info(f"Retrieved {len(visualizer_poses)} poses from visualizer")
        
        # Convert to planning targets
        planning_targets = self.convert_multiple_poses(visualizer_poses)
        
        return planning_targets
    
    def create_executor(self, execution_mode: str = "blend", chunk_size: int = 8) -> PlanningDynamicExecutor:
        """
        Create and initialize a planning dynamic executor.
        
        Args:
            execution_mode: "blend" or "point" execution mode
            chunk_size: Number of waypoints per chunk
            
        Returns:
            Initialized PlanningDynamicExecutor
        """
        if self.executor is None:
            self.executor = PlanningDynamicExecutor(
                robot_ip=self.robot_ip,
                execution_mode=execution_mode,
                operation_mode=self.operation_mode,
                chunk_size=chunk_size
            )
            
            # Initialize the executor
            if self.executor.initialize():
                logger.info("Planning dynamic executor initialized successfully")
            else:
                logger.warning("Executor initialization failed - limited functionality")
        
        return self.executor
    
    def plan_and_execute_poses(self, planning_targets: List[PlanningTarget], 
                             execution_mode: str = "blend") -> List[bool]:
        """
        Plan and execute motion to multiple poses.
        
        Args:
            planning_targets: List of PlanningTarget objects
            execution_mode: "blend" or "point" execution mode
            
        Returns:
            List of success flags for each motion
        """
        if not planning_targets:
            logger.warning("No planning targets provided")
            return []
        
        # Create executor if needed
        executor = self.create_executor(execution_mode)
        
        results = []
        for i, target in enumerate(planning_targets):
            logger.info(f"Executing motion {i+1}/{len(planning_targets)} to {target.tcp_position_mm}")
            
            try:
                success = executor.plan_and_execute_motion(target)
                results.append(success)
                
                if success:
                    logger.info(f"Motion {i+1} completed successfully")
                else:
                    logger.error(f"Motion {i+1} failed")
                    
            except Exception as e:
                logger.error(f"Error executing motion {i+1}: {e}")
                results.append(False)
        
        success_count = sum(results)
        logger.info(f"Completed {success_count}/{len(results)} motions successfully")
        
        return results
    
    def run_complete_workflow(self, execution_mode: str = "blend") -> Tuple[List[PlanningTarget], List[bool]]:
        """
        Run the complete workflow: pose selection → motion planning → execution.
        
        Args:
            execution_mode: "blend" or "point" execution mode
            
        Returns:
            Tuple of (planning_targets, execution_results)
        """
        logger.info("Starting complete pose-to-motion workflow...")
        
        # Step 1: Interactive pose selection
        planning_targets = self.run_interactive_pose_selection()
        
        if not planning_targets:
            logger.warning("Workflow cancelled - no poses selected")
            return [], []
        
        # Step 2: Plan and execute motions
        execution_results = self.plan_and_execute_poses(planning_targets, execution_mode)
        
        # Step 3: Summary
        success_count = sum(execution_results)
        logger.info(f"Workflow complete: {success_count}/{len(planning_targets)} motions successful")
        
        return planning_targets, execution_results
    
    def get_pose_summary(self, planning_target: PlanningTarget) -> str:
        """
        Get a formatted summary of a planning target.
        
        Args:
            planning_target: PlanningTarget object
            
        Returns:
            Formatted string with pose information
        """
        pos = planning_target.tcp_position_mm
        rot = planning_target.tcp_rotation_deg
        
        return (f"Position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}] mm, "
                f"Rotation: [{rot[0]:.1f}, {rot[1]:.1f}, {rot[2]:.1f}] deg")
    
    def shutdown(self):
        """Clean shutdown of all components."""
        if self.executor:
            self.executor.shutdown()
            logger.info("Executor shut down")
        
        if self.visualizer:
            self.visualizer.clear_poses()
            logger.info("Visualizer cleared")
        
        logger.info("Pose integration bridge shut down")

def create_integrated_workflow(robot_ip: str = "192.168.0.10", 
                             operation_mode: str = "simulation") -> PoseIntegrationBridge:
    """
    Convenience function to create an integrated workflow bridge.
    
    Args:
        robot_ip: Robot controller IP address
        operation_mode: "simulation" or "real"
        
    Returns:
        Configured PoseIntegrationBridge
    """
    return PoseIntegrationBridge(robot_ip, operation_mode)

# Example usage demonstration
def demo_integration():
    """Demonstrate the integration bridge functionality."""
    print("POSE INTEGRATION BRIDGE DEMO")
    print("="*50)
    
    # Create integration bridge
    bridge = create_integrated_workflow(
        robot_ip="192.168.0.10",
        operation_mode="simulation"
    )
    
    try:
        # Run complete workflow
        targets, results = bridge.run_complete_workflow(execution_mode="blend")
        
        # Display results
        print(f"\nWORKFLOW RESULTS:")
        print(f"Poses selected: {len(targets)}")
        print(f"Motions successful: {sum(results)}")
        
        for i, (target, success) in enumerate(zip(targets, results)):
            status = "SUCCESS" if success else "FAILED"
            print(f"  {status} Pose {i+1}: {bridge.get_pose_summary(target)}")
    
    except KeyboardInterrupt:
        print("\nDemo cancelled by user")
    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        bridge.shutdown()

if __name__ == "__main__":
    demo_integration()