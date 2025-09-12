#!/usr/bin/env python3
"""
Clean Pick and Place Example

This example demonstrates a complete pick and place operation with:
- Approach and retreat sequences for safety
- Retry mechanisms for robust operation
- Real-world units (mm, degrees) throughout
- Direct robot API compatibility
- Error handling and recovery

The pick and place sequence includes:
1. Approach above pick location
2. Move down to pick
3. Retreat after picking  
4. Approach above place location
5. Move down to place
6. Final retreat

Each step includes retry logic and safety checks.
"""

import sys
import os
import numpy as np
import time
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'kinematics', 'src'))

from clean_robot_interface import CleanRobotMotionPlanner, RobotPose, RobotMotionPlan

class PickPlaceStatus(Enum):
    """Pick and place operation status."""
    SUCCESS = "success"
    FAILED_APPROACH_PICK = "failed_approach_pick"
    FAILED_PICK = "failed_pick"
    FAILED_RETREAT_PICK = "failed_retreat_pick"
    FAILED_APPROACH_PLACE = "failed_approach_place"
    FAILED_PLACE = "failed_place"
    FAILED_RETREAT_PLACE = "failed_retreat_place"
    FAILED_RETURN_HOME = "failed_return_home"
    INVALID_INPUT = "invalid_input"
    PLANNING_ERROR = "planning_error"

@dataclass
class PickPlaceConfig:
    """Configuration for pick and place operation."""
    approach_height_mm: float = 50.0        # Height above pick/place locations
    retreat_height_mm: float = 80.0         # Height for safe retreat  
    speed_approach_percent: float = 30.0    # Speed for approach moves
    speed_pick_place_percent: float = 15.0  # Speed for pick/place moves
    speed_retreat_percent: float = 40.0     # Speed for retreat moves
    max_retries: int = 3                    # Maximum retry attempts per step
    retry_delay_sec: float = 0.5            # Delay between retries
    blend_radius_mm: float = 3.0            # Blend radius for smooth motion
    
    # Safety limits
    min_z_mm: float = 60.0                  # Minimum Z height (table level)
    max_z_mm: float = 1000.0               # Maximum Z height
    workspace_radius_mm: float = 600.0      # Workspace radius from center

@dataclass
class PickPlaceResult:
    """Result of pick and place operation."""
    status: PickPlaceStatus
    success: bool
    total_time_sec: float
    steps_completed: List[str]
    error_message: str = ""
    robot_programs: List[str] = None
    
    def __post_init__(self):
        if self.robot_programs is None:
            self.robot_programs = []

class PickAndPlaceExecutor:
    """
    Complete pick and place executor with retry logic and safety features.
    
    Features:
    - Multi-step approach/retreat sequences
    - Configurable retry mechanisms  
    - Safety validation at each step
    - Real-world units (mm, degrees)
    - Robot API compatible output
    """
    
    def __init__(self, config: Optional[PickPlaceConfig] = None):
        """Initialize pick and place executor."""
        self.planner = CleanRobotMotionPlanner()
        self.config = config or PickPlaceConfig()
        
    def validate_position(self, position_mm: List[float], name: str) -> Tuple[bool, str]:
        """Validate position is within safe workspace."""
        x, y, z = position_mm
        
        # Check Z limits
        if z < self.config.min_z_mm:
            return False, f"{name} Z ({z:.1f}mm) below minimum ({self.config.min_z_mm}mm)"
        if z > self.config.max_z_mm:
            return False, f"{name} Z ({z:.1f}mm) above maximum ({self.config.max_z_mm}mm)"
        
        # Check workspace radius
        radius = np.sqrt(x**2 + y**2)
        if radius > self.config.workspace_radius_mm:
            return False, f"{name} radius ({radius:.1f}mm) exceeds workspace ({self.config.workspace_radius_mm}mm)"
        
        return True, "Valid"
    
    def plan_motion_with_retry(self, 
                              current_joints: List[float],
                              target_pos: List[float], 
                              target_rot: List[float],
                              step_name: str) -> Tuple[bool, RobotMotionPlan]:
        """Plan motion with retry mechanism."""
        for attempt in range(self.config.max_retries):
            try:
                plan = self.planner.plan_motion(current_joints, target_pos, target_rot)
                
                if plan.success:
                    if attempt > 0:
                        print(f"  ✅ {step_name} succeeded on attempt {attempt + 1}")
                    return True, plan
                else:
                    print(f"  ⚠️ {step_name} attempt {attempt + 1} failed: {plan.error_message}")
                    
            except Exception as e:
                print(f"  ❌ {step_name} attempt {attempt + 1} exception: {str(e)}")
            
            # Wait before retry (except last attempt)
            if attempt < self.config.max_retries - 1:
                time.sleep(self.config.retry_delay_sec)
        
        return False, RobotMotionPlan([], 0.0, 0.0, False, f"All {self.config.max_retries} attempts failed")
    
    def execute_pick_and_place(self,
                              current_joints_deg: List[float],
                              current_tcp_position_mm: List[float],
                              current_tcp_orientation_deg: List[float],
                              pick_position_mm: List[float],
                              pick_orientation_deg: List[float],
                              place_position_mm: List[float],
                              place_orientation_deg: List[float],
                              return_home: bool = True) -> PickPlaceResult:
        """
        Execute complete pick and place operation.
        
        Args:
            current_joints_deg: Current robot joint angles in degrees
            current_tcp_position_mm: Current TCP position [x, y, z] in mm
            current_tcp_orientation_deg: Current TCP orientation [rx, ry, rz] in degrees
            pick_position_mm: Pick target position [x, y, z] in mm
            pick_orientation_deg: Pick target orientation [rx, ry, rz] in degrees  
            place_position_mm: Place target position [x, y, z] in mm
            place_orientation_deg: Place target orientation [rx, ry, rz] in degrees
            return_home: Whether to return to home position after place
            
        Returns:
            PickPlaceResult with execution details and robot programs
        """
        start_time = time.time()
        steps_completed = []
        robot_programs = []
        
        print("🤖 PICK AND PLACE EXECUTION")
        print("=" * 30)
        
        # Validate inputs
        valid_current, msg_current = self.validate_position(current_tcp_position_mm, "Current TCP position")
        if not valid_current:
            return PickPlaceResult(PickPlaceStatus.INVALID_INPUT, False, 0.0, [], msg_current)
        
        valid_pick, msg_pick = self.validate_position(pick_position_mm, "Pick target position")
        if not valid_pick:
            return PickPlaceResult(PickPlaceStatus.INVALID_INPUT, False, 0.0, [], msg_pick)
        
        valid_place, msg_place = self.validate_position(place_position_mm, "Place target position") 
        if not valid_place:
            return PickPlaceResult(PickPlaceStatus.INVALID_INPUT, False, 0.0, [], msg_place)
        
        print(f"✅ Input validation passed")
        print(f"Current TCP: {current_tcp_position_mm} mm, {current_tcp_orientation_deg}°")
        print(f"Pick target: {pick_position_mm} mm, {pick_orientation_deg}°")  
        print(f"Place target: {place_position_mm} mm, {place_orientation_deg}°")
        
        current_joints = current_joints_deg.copy()
        
        try:
            # Step 1: Approach pick position
            print(f"\n📍 STEP 1: Approach pick position")
            approach_pick_pos = pick_position_mm.copy()
            approach_pick_pos[2] += self.config.approach_height_mm
            
            success, plan = self.plan_motion_with_retry(
                current_joints, approach_pick_pos, pick_orientation_deg, "Approach pick"
            )
            
            if not success:
                return PickPlaceResult(PickPlaceStatus.FAILED_APPROACH_PICK, False, 
                                     time.time() - start_time, steps_completed, plan.error_message)
            
            steps_completed.append("approach_pick")
            robot_programs.append(plan.generate_robot_program(self.config.speed_approach_percent))
            current_joints = plan.waypoints[-1].joints_deg
            print(f"  ✅ Approach pick: {len(plan.waypoints)} waypoints, {plan.execution_time_sec:.2f}s")
            
            # Step 2: Move to pick position
            print(f"\n🎯 STEP 2: Move to pick position")
            success, plan = self.plan_motion_with_retry(
                current_joints, pick_position_mm, pick_orientation_deg, "Pick"
            )
            
            if not success:
                return PickPlaceResult(PickPlaceStatus.FAILED_PICK, False,
                                     time.time() - start_time, steps_completed, plan.error_message)
            
            steps_completed.append("pick")
            robot_programs.append(plan.generate_robot_program(self.config.speed_pick_place_percent))
            current_joints = plan.waypoints[-1].joints_deg
            print(f"  ✅ Pick: {len(plan.waypoints)} waypoints, {plan.execution_time_sec:.2f}s")
            
            # Step 3: Retreat from pick
            print(f"\n⬆️ STEP 3: Retreat from pick")
            retreat_pick_pos = pick_position_mm.copy()
            retreat_pick_pos[2] += self.config.retreat_height_mm
            
            success, plan = self.plan_motion_with_retry(
                current_joints, retreat_pick_pos, pick_orientation_deg, "Retreat pick"
            )
            
            if not success:
                return PickPlaceResult(PickPlaceStatus.FAILED_RETREAT_PICK, False,
                                     time.time() - start_time, steps_completed, plan.error_message)
            
            steps_completed.append("retreat_pick")
            robot_programs.append(plan.generate_robot_program(self.config.speed_retreat_percent))
            current_joints = plan.waypoints[-1].joints_deg
            print(f"  ✅ Retreat pick: {len(plan.waypoints)} waypoints, {plan.execution_time_sec:.2f}s")
            
            # Step 4: Approach place position
            print(f"\n📍 STEP 4: Approach place position")
            approach_place_pos = place_position_mm.copy()
            approach_place_pos[2] += self.config.approach_height_mm
            
            success, plan = self.plan_motion_with_retry(
                current_joints, approach_place_pos, place_orientation_deg, "Approach place"
            )
            
            if not success:
                return PickPlaceResult(PickPlaceStatus.FAILED_APPROACH_PLACE, False,
                                     time.time() - start_time, steps_completed, plan.error_message)
            
            steps_completed.append("approach_place")
            robot_programs.append(plan.generate_robot_program(self.config.speed_approach_percent))
            current_joints = plan.waypoints[-1].joints_deg
            print(f"  ✅ Approach place: {len(plan.waypoints)} waypoints, {plan.execution_time_sec:.2f}s")
            
            # Step 5: Move to place position
            print(f"\n🎯 STEP 5: Move to place position")
            success, plan = self.plan_motion_with_retry(
                current_joints, place_position_mm, place_orientation_deg, "Place"
            )
            
            if not success:
                return PickPlaceResult(PickPlaceStatus.FAILED_PLACE, False,
                                     time.time() - start_time, steps_completed, plan.error_message)
            
            steps_completed.append("place")
            robot_programs.append(plan.generate_robot_program(self.config.speed_pick_place_percent))
            current_joints = plan.waypoints[-1].joints_deg
            print(f"  ✅ Place: {len(plan.waypoints)} waypoints, {plan.execution_time_sec:.2f}s")
            
            # Step 6: Retreat from place
            print(f"\n⬆️ STEP 6: Retreat from place")
            retreat_place_pos = place_position_mm.copy()
            retreat_place_pos[2] += self.config.retreat_height_mm
            
            success, plan = self.plan_motion_with_retry(
                current_joints, retreat_place_pos, place_orientation_deg, "Retreat place"
            )
            
            if not success:
                return PickPlaceResult(PickPlaceStatus.FAILED_RETREAT_PLACE, False,
                                     time.time() - start_time, steps_completed, plan.error_message)
            
            steps_completed.append("retreat_place")
            robot_programs.append(plan.generate_robot_program(self.config.speed_retreat_percent))
            current_joints = plan.waypoints[-1].joints_deg
            print(f"  ✅ Retreat place: {len(plan.waypoints)} waypoints, {plan.execution_time_sec:.2f}s")
            
            # Step 7: Return home (optional)
            if return_home:
                print(f"\n🏠 STEP 7: Return to home position")
                # Return to original TCP position and orientation
                success, plan = self.plan_motion_with_retry(
                    current_joints, current_tcp_position_mm, current_tcp_orientation_deg, "Return home"
                )
                
                if not success:
                    return PickPlaceResult(PickPlaceStatus.FAILED_RETURN_HOME, False,
                                         time.time() - start_time, steps_completed, plan.error_message)
                
                steps_completed.append("return_home")
                robot_programs.append(plan.generate_robot_program(self.config.speed_approach_percent))
                print(f"  ✅ Return home: {len(plan.waypoints)} waypoints, {plan.execution_time_sec:.2f}s")
            
            total_time = time.time() - start_time
            print(f"\n✅ PICK AND PLACE COMPLETE!")
            print(f"Total steps: {len(steps_completed)}")
            print(f"Total time: {total_time:.2f} seconds")
            
            return PickPlaceResult(
                status=PickPlaceStatus.SUCCESS,
                success=True,
                total_time_sec=total_time,
                steps_completed=steps_completed,
                robot_programs=robot_programs
            )
            
        except Exception as e:
            return PickPlaceResult(
                status=PickPlaceStatus.PLANNING_ERROR,
                success=False,
                total_time_sec=time.time() - start_time,
                steps_completed=steps_completed,
                error_message=f"Unexpected error: {str(e)}",
                robot_programs=robot_programs
            )

def demo_pick_and_place():
    """Demonstrate pick and place operation."""
    print("🔄 PICK AND PLACE DEMONSTRATION")
    print("=" * 35)
    
    # Configuration
    config = PickPlaceConfig(
        approach_height_mm=60.0,
        retreat_height_mm=100.0, 
        speed_approach_percent=25.0,
        speed_pick_place_percent=10.0,
        max_retries=2
    )
    
    # Initialize executor
    executor = PickAndPlaceExecutor(config)
    print("✅ Pick and place executor initialized")
    
    # Define starting point (known joints and TCP pose)
    current_joints = [0.0, -20.0, 30.0, 0.0, -10.0, 0.0]  # degrees
    
    # Get current TCP pose from joints (for validation and home return)
    current_tcp_pose = executor.planner.get_current_pose_from_joints(current_joints)
    current_tcp_pos = current_tcp_pose.position_mm
    current_tcp_rot = current_tcp_pose.orientation_deg
    
    # Define target poses (only position and orientation in mm/degrees)
    # Pick target (above table)
    pick_pos = [250.0, 150.0, 80.0]        # mm (20mm above 60mm table)
    pick_rot = [180.0, 0.0, 0.0]           # gripper pointing down
    
    # Place target 
    place_pos = [-200.0, 200.0, 100.0]     # mm (different location)
    place_rot = [180.0, 0.0, 45.0]         # gripper down, rotated 45°
    
    print(f"\n📋 Operation Details:")
    print(f"Current joints: {current_joints} degrees")
    print(f"Current TCP: {[round(x,1) for x in current_tcp_pos]} mm, {[round(x,1) for x in current_tcp_rot]}°")
    print(f"Pick target: {pick_pos} mm, {pick_rot}°")
    print(f"Place target: {place_pos} mm, {place_rot}°")
    print(f"Config: {config.approach_height_mm}mm approach, {config.retreat_height_mm}mm retreat")
    
    # Execute pick and place with current TCP pose and target poses
    result = executor.execute_pick_and_place(
        current_joints, current_tcp_pos, current_tcp_rot,
        pick_pos, pick_rot, place_pos, place_rot, return_home=True
    )
    
    # Show results
    print(f"\n📊 RESULTS:")
    print(f"Status: {result.status.value}")
    print(f"Success: {'✅' if result.success else '❌'}")
    print(f"Steps completed: {len(result.steps_completed)}/{7}")  
    print(f"Total time: {result.total_time_sec:.2f} seconds")
    
    if result.success:
        print(f"\n🤖 ROBOT PROGRAMS GENERATED:")
        for i, program in enumerate(result.robot_programs):
            step_name = result.steps_completed[i] if i < len(result.steps_completed) else f"step_{i+1}"
            print(f"\n--- {step_name.upper()} PROGRAM ---")
            lines = program.split('\\n')
            for line in lines[:8]:  # Show first 8 lines
                print(line)
            if len(lines) > 8:
                print("...")
    else:
        print(f"❌ Error: {result.error_message}")
    
    print(f"\n✅ Pick and place demonstration complete!")

if __name__ == "__main__":
    demo_pick_and_place()
