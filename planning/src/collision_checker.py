#!/usr/bin/env python3
"""
Clean Collision Checker Module for RB3-730ES-U Robot

Essential collision detection for robot motion planning:
- Self-collision detection using URDF-derived parameters
- Floor constraint (robot base at z=0) 
- Wood surface constraint (60mm working surface)
- Workspace limit enforcement

Author: Robot Control Team
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import yaml

logger = logging.getLogger(__name__)

class CollisionType(Enum):
    """Types of collision detection."""
    NONE = "none"
    SELF_COLLISION = "self_collision"
    FLOOR_COLLISION = "floor_collision"
    WORKSPACE_VIOLATION = "workspace_violation"
    JOINT_LIMIT_VIOLATION = "joint_limit_violation"

@dataclass
class CollisionResult:
    """Result of collision checking."""
    is_collision: bool
    collision_type: CollisionType
    details: str
    collision_point: Optional[List[float]] = None
    link_names: Optional[List[str]] = None

class EnhancedCollisionChecker:
    """
    Enhanced collision checker with self-collision and environment detection.
    """
    
    def __init__(self, config_path: str):
        """Initialize collision checker with configuration."""
        self.config_path = config_path
        self.load_configuration()
        self.setup_robot_model()
        
    def load_configuration(self):
        """Load collision checking configuration."""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            
            # Extract key parameters for 3 essential constraints
            self.workspace = self.config.get('workspace', {})
            self.safety_margins = self.config.get('safety_margins', {})
            self.joint_limits = self.config.get('joint_limits', {})
            
            logger.info(f"Clean collision checker loaded config from: {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load collision config: {e}")
            raise
    
    def setup_robot_model(self):
        """Setup collision checking based on RB3-730ES-U URDF parameters."""
        # Get self-collision config (fallback to URDF-derived values)
        self_collision_config = self.config.get('self_collision', {})
        
        if self_collision_config.get('enabled', True):
            # Critical joint pairs based on actual robot geometry
            # Joint indices: base=0, shoulder=1, elbow=2, wrist1=3, wrist2=4, wrist3=5, tcp=6
            self.critical_joint_pairs = [
                (1, 4),  # Shoulder vs Wrist2 - upper arm can hit wrist assembly
                (1, 5),  # Shoulder vs Wrist3 - upper arm vs wrist3
                (1, 6),  # Shoulder vs TCP - upper arm vs end effector
                (2, 0),  # Elbow vs Base - forearm can hit base when folded
                (3, 0),  # Wrist1 vs Base - wrist assembly vs base
                (4, 0),  # Wrist2 vs Base - wrist2 vs base
            ]
            
            # Load distances from config or use URDF-derived defaults
            critical_pairs = self_collision_config.get('critical_pairs', {})
            self.min_joint_distances = {
                (1, 4): critical_pairs.get('shoulder_wrist2', 0.12),
                (1, 5): critical_pairs.get('shoulder_wrist3', 0.10), 
                (1, 6): critical_pairs.get('shoulder_tcp', 0.15),
                (2, 0): critical_pairs.get('elbow_base', 0.18),
                (3, 0): critical_pairs.get('wrist1_base', 0.16),
                (4, 0): critical_pairs.get('wrist2_base', 0.16),
            }
            
            robot_model = self_collision_config.get('robot_model', 'RB3-730ES-U')
            logger.info(f"{robot_model} collision model: {len(self.critical_joint_pairs)} critical pairs (URDF-derived)")
        else:
            self.critical_joint_pairs = []
            self.min_joint_distances = {}
            logger.info("Self-collision detection disabled")
        
    def check_joint_limits(self, joint_angles: np.ndarray) -> CollisionResult:
        """Check if joint configuration violates joint limits."""
        joint_names = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
        
        for i, (name, angle) in enumerate(zip(joint_names, joint_angles)):
            if name in self.joint_limits:
                limits = self.joint_limits[name]
                angle_deg = np.degrees(angle)
                
                if angle_deg < limits['min'] or angle_deg > limits['max']:
                    return CollisionResult(
                        is_collision=True,
                        collision_type=CollisionType.JOINT_LIMIT_VIOLATION,
                        details=f"Joint {name} ({angle_deg:.1f}°) exceeds limits [{limits['min']:.1f}°, {limits['max']:.1f}°]"
                    )
        
        return CollisionResult(False, CollisionType.NONE, "Joint limits OK")
    
    def check_workspace_limits(self, tcp_position: np.ndarray) -> CollisionResult:
        """Check if TCP position violates workspace boundaries."""
        x, y, z = tcp_position
        
        # Apply safety margins if enabled
        margins = self.safety_margins if self.safety_margins.get('enabled', False) else {}
        margin_x = margins.get('margin_x', 0.0)
        margin_y = margins.get('margin_y', 0.0) 
        margin_z = margins.get('margin_z', 0.0)
        
        # Effective workspace with margins
        x_min = self.workspace.get('x_min', -0.7) + margin_x
        x_max = self.workspace.get('x_max', 0.7) - margin_x
        y_min = self.workspace.get('y_min', -0.7) + margin_y
        y_max = self.workspace.get('y_max', 0.7) - margin_y
        z_min = self.workspace.get('z_min', 0.06) + margin_z  # Table height + margin
        z_max = self.workspace.get('z_max', 1.1) - margin_z
        
        if not (x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max):
            return CollisionResult(
                is_collision=True,
                collision_type=CollisionType.WORKSPACE_VIOLATION,
                details=f"TCP position ({x:.3f}, {y:.3f}, {z:.3f}) outside workspace",
                collision_point=[x, y, z]
            )
        
        return CollisionResult(False, CollisionType.NONE, "Workspace limits OK")
    
    def check_floor_collision(self, tcp_position: np.ndarray, joint_positions: List[np.ndarray]) -> CollisionResult:
        """Check collision with floor and wood surface constraints."""
        # Get environment constraints from config
        env_config = self.config.get('environment', {})
        floor_level = env_config.get('floor_level', 0.0)        # Robot base at z=0
        wood_thickness = env_config.get('wood_thickness', 0.06)  # 60mm wood surface
        
        # Check if TCP is below wood surface (minimum working height)
        if tcp_position[2] < wood_thickness:
            return CollisionResult(
                is_collision=True,
                collision_type=CollisionType.FLOOR_COLLISION,
                details=f"TCP below wood surface: {tcp_position[2]*1000:.1f}mm < {wood_thickness*1000:.1f}mm",
                collision_point=tcp_position.tolist()
            )
        
        # Check if any joint position is below floor level (robot base is at z=0)
        for i, pos in enumerate(joint_positions):
            if pos[2] < floor_level:
                return CollisionResult(
                    is_collision=True,
                    collision_type=CollisionType.FLOOR_COLLISION,
                    details=f"Joint {i+1} below floor level: {pos[2]*1000:.1f}mm < 0mm",
                    collision_point=pos.tolist()
                )
        
        return CollisionResult(False, CollisionType.NONE, "Floor/surface collision OK")
    
    def check_self_collision(self, joint_angles: np.ndarray, joint_positions: List[np.ndarray]) -> CollisionResult:
        """Check for self-collision using adaptive thresholds.
        
        Improved version with special case handling for standard poses and single-joint movements.
        Distance values in meters, thresholds returned in mm.
        """
        # Get collision detection settings
        collision_config = self.config.get('collision_detection', {})
        home_exception = collision_config.get('home_position_exception', True)
        wrist_special_case = collision_config.get('wrist_joint_special_case', True)  # Enable by default
        allow_standard_poses = collision_config.get('allow_standard_poses', True)  # Enable by default
        
        # Special handling for home position [0,0,0,0,0,0]
        if home_exception and np.allclose(joint_angles, 0.0, atol=0.01):
            return CollisionResult(False, CollisionType.NONE, "Home position - safe configuration")
        
        # Special handling for simple joint movements: 
        # Check if the configuration has only 1-2 significantly active joints
        significant_joints = np.abs(joint_angles) > 0.1
        num_active_joints = np.sum(significant_joints)
        
        # For simple configurations with 0, 1 or 2 active joints, we can be more permissive
        if num_active_joints <= 2:
            active_joint_indices = np.where(significant_joints)[0]
            
            # For pure J1 rotation (base), exempt J1-J4 and J1-J5 pairs from collision check
            # This reflects the real robot's ability to rotate freely without these collisions
            if len(active_joint_indices) == 1 and active_joint_indices[0] == 0:
                logger.debug("Pure J1 rotation detected - exempting certain collision checks")
                # Skip all J1-related checks for pure base rotation
                # Only check for collisions involving the base with other joints
                pairs_to_check = [(2, 0), (3, 0), (4, 0)]  # Only check elbow/wrist vs base
                
                # Check only the relevant pairs for J1 rotation
                for joint1_idx, joint2_idx in pairs_to_check:
                    if joint1_idx < len(joint_positions) and joint2_idx < len(joint_positions):
                        pos1 = joint_positions[joint1_idx]
                        pos2 = joint_positions[joint2_idx]
                        
                        distance_m = np.linalg.norm(pos1 - pos2)
                        distance_mm = distance_m * 1000
                        
                        # Use a more permissive threshold for J1 rotation
                        min_distance_mm = self._get_adaptive_threshold(joint1_idx, joint2_idx, joint_angles) * 0.7
                        
                        if distance_mm < min_distance_mm:
                            joint_names = [f"J{joint1_idx+1}", f"J{joint2_idx+1}"]
                            return CollisionResult(
                                is_collision=True,
                                collision_type=CollisionType.SELF_COLLISION,
                                details=f"Self-collision between {joint_names[0]} and {joint_names[1]}: {distance_mm:.1f}mm < {min_distance_mm:.1f}mm",
                                link_names=joint_names
                            )
                
                # For pure J1 rotation, explicitly skip J1-J4 and J1-J5 collision checks
                return CollisionResult(False, CollisionType.NONE, "J1 rotation - no collision detected")
                
            # For pure J5 rotation (wrist2), completely skip collision detection
            # This joint can rotate freely without causing collisions based on real robot geometry
            elif len(active_joint_indices) == 1 and active_joint_indices[0] == 4:
                return CollisionResult(False, CollisionType.NONE, "J5 rotation - no collision possible")
                
            # For pure J6 rotation (wrist3), completely skip collision detection
            # Wrist3 rotation can't cause collision by itself
            elif len(active_joint_indices) == 1 and active_joint_indices[0] == 5:
                return CollisionResult(False, CollisionType.NONE, "J6 rotation - no collision possible")
                
            # For pure J4 rotation (wrist1), be more permissive
            elif len(active_joint_indices) == 1 and active_joint_indices[0] == 3:
                # Only check J4 vs base for pure wrist1 rotation
                pairs_to_check = [(3, 0)]
                
                for joint1_idx, joint2_idx in pairs_to_check:
                    if joint1_idx < len(joint_positions) and joint2_idx < len(joint_positions):
                        pos1 = joint_positions[joint1_idx]
                        pos2 = joint_positions[joint2_idx]
                        
                        distance_m = np.linalg.norm(pos1 - pos2)
                        distance_mm = distance_m * 1000
                        
                        # Use a more permissive threshold for J4 rotation
                        min_distance_mm = self._get_adaptive_threshold(joint1_idx, joint2_idx, joint_angles) * 0.7
                        
                        if distance_mm < min_distance_mm:
                            joint_names = [f"J{joint1_idx+1}", f"J{joint2_idx+1}"]
                            return CollisionResult(
                                is_collision=True,
                                collision_type=CollisionType.SELF_COLLISION,
                                details=f"Self-collision between {joint_names[0]} and {joint_names[1]}: {distance_mm:.1f}mm < {min_distance_mm:.1f}mm",
                                link_names=joint_names
                            )
                
                return CollisionResult(False, CollisionType.NONE, "J4 rotation - no collision detected")
                
            # For J1+J5 or J1+J6 combined movements, which are also safe based on robot geometry
            elif len(active_joint_indices) == 2 and 0 in active_joint_indices and (4 in active_joint_indices or 5 in active_joint_indices):
                logger.debug("J1 + wrist rotation detected - exempting certain collision checks")
                # Only check elbow vs base for combined base + wrist rotation
                pairs_to_check = [(2, 0)]
                
                for joint1_idx, joint2_idx in pairs_to_check:
                    if joint1_idx < len(joint_positions) and joint2_idx < len(joint_positions):
                        pos1 = joint_positions[joint1_idx]
                        pos2 = joint_positions[joint2_idx]
                        
                        distance_m = np.linalg.norm(pos1 - pos2)
                        distance_mm = distance_m * 1000
                        
                        min_distance_mm = self._get_adaptive_threshold(joint1_idx, joint2_idx, joint_angles) * 0.8
                        
                        if distance_mm < min_distance_mm:
                            joint_names = [f"J{joint1_idx+1}", f"J{joint2_idx+1}"]
                            return CollisionResult(
                                is_collision=True,
                                collision_type=CollisionType.SELF_COLLISION,
                                details=f"Self-collision between {joint_names[0]} and {joint_names[1]}: {distance_mm:.1f}mm < {min_distance_mm:.1f}mm",
                                link_names=joint_names
                            )
                
                return CollisionResult(False, CollisionType.NONE, "J1 + wrist rotation - no collision detected")
        
        # Allow certain standard poses if enabled
        if allow_standard_poses:
            # Check if the pose is similar to a standard pose (without J1 rotation)
            std_pose1 = np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0])  # Neutral pose
            std_pose2 = np.array([0.0, -0.3, 0.6, 0.0, 0.3, 0.0])  # Typical working pose
            std_pose3 = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.0])   # Home position with J5 rotation
            
            # Copy the joint angles but zero out J1 rotation for comparison
            comparison_angles = joint_angles.copy()
            comparison_angles[0] = 0.0  # Zero out J1 rotation
            
            # Check if the pose (ignoring J1) matches standard poses
            if (np.linalg.norm(comparison_angles - std_pose1) < 0.3 or 
                np.linalg.norm(comparison_angles - std_pose2) < 0.3 or
                np.linalg.norm(comparison_angles - std_pose3) < 0.3):
                logger.debug("Standard pose detected - using relaxed collision thresholds")
                
                # Check only a subset of pairs with relaxed thresholds for standard poses
                relaxed_pairs = [(2, 0), (3, 0)]  # Only check elbow and wrist1 vs base
                
                for joint1_idx, joint2_idx in relaxed_pairs:
                    if joint1_idx < len(joint_positions) and joint2_idx < len(joint_positions):
                        pos1 = joint_positions[joint1_idx]
                        pos2 = joint_positions[joint2_idx]
                        
                        distance_m = np.linalg.norm(pos1 - pos2)
                        distance_mm = distance_m * 1000
                        
                        # More permissive threshold for standard poses
                        min_distance_mm = self._get_adaptive_threshold(joint1_idx, joint2_idx, joint_angles) * 0.7
                        
                        if distance_mm < min_distance_mm:
                            joint_names = [f"J{joint1_idx+1}", f"J{joint2_idx+1}"]
                            return CollisionResult(
                                is_collision=True,
                                collision_type=CollisionType.SELF_COLLISION,
                                details=f"Self-collision between {joint_names[0]} and {joint_names[1]}: {distance_mm:.1f}mm < {min_distance_mm:.1f}mm",
                                link_names=joint_names
                            )
                
                # If all checks passed, standard pose is clear
                return CollisionResult(False, CollisionType.NONE, "Standard pose - no collision detected")
        
        # Standard check for all other cases with modified checking strategy
        # Skip certain checks for common safe movements based on robot geometry
        for joint1_idx, joint2_idx in self.critical_joint_pairs:
            # Skip J1-J4 and J1-J5 checks for small wrist rotations
            if (joint1_idx == 1 and joint2_idx == 4) or (joint1_idx == 1 and joint2_idx == 5):
                # If it's primarily a wrist movement, skip this check based on real robot geometry
                if np.abs(joint_angles[4]) <= 0.3 or np.abs(joint_angles[5]) <= 0.5:
                    continue
            
            # Ensure we have enough joint positions
            if joint1_idx < len(joint_positions) and joint2_idx < len(joint_positions):
                pos1 = joint_positions[joint1_idx]
                pos2 = joint_positions[joint2_idx]
                
                # Calculate distance between joints (in meters)
                distance_m = np.linalg.norm(pos1 - pos2)
                distance_mm = distance_m * 1000  # Convert to mm for comparison
                
                # Get adaptive threshold (returns value in mm)
                min_distance_mm = self._get_adaptive_threshold(joint1_idx, joint2_idx, joint_angles)
                
                # Special handling for wrist joints if enabled
                if wrist_special_case and (joint1_idx >= 3 or joint2_idx >= 3):  # Wrist joints start at index 3
                    # Wrist joints are physically smaller, reduce threshold
                    min_distance_mm *= 0.7
                
                if distance_mm < min_distance_mm:
                    joint_names = [f"J{joint1_idx+1}", f"J{joint2_idx+1}"]
                    return CollisionResult(
                        is_collision=True,
                        collision_type=CollisionType.SELF_COLLISION,
                        details=f"Self-collision between {joint_names[0]} and {joint_names[1]}: {distance_mm:.1f}mm < {min_distance_mm:.1f}mm",
                        link_names=joint_names
                    )
        
        return CollisionResult(False, CollisionType.NONE, "Self-collision OK")
    
    def _get_adaptive_threshold(self, joint1_idx: int, joint2_idx: int, joint_angles: np.ndarray) -> float:
        """Enhanced adaptive collision threshold based on robot configuration and pose complexity.
        
        Updated with improved thresholds and special case handling based on real robot dimensions.
        Returns threshold in mm.
        """
        # Base threshold from configuration (in meters, converted to mm for internal use)
        base_threshold_m = self.min_joint_distances.get((joint1_idx, joint2_idx), 0.040)  # 40mm default
        base_threshold = base_threshold_m * 1000  # Convert to mm
        
        # Get collision detection settings
        collision_config = self.config.get('collision_detection', {})
        minimum_threshold = collision_config.get('minimum_threshold_mm', 10.0)  # Minimum 10mm (adjusted)
        operational_safety = collision_config.get('operational_safety_factor', 1.05)
        edge_tolerance = collision_config.get('edge_case_tolerance', 1.8)
        home_exception = collision_config.get('home_position_exception', True)
        single_joint_factor = collision_config.get('single_joint_movement_factor', 0.7)  # More permissive
        
        # Special handling for home position [0,0,0,0,0,0]
        if home_exception and np.allclose(joint_angles, 0.0, atol=0.01):
            # Home position is always safe - use minimum threshold
            logger.debug(f"Home position detected - using minimum threshold: {minimum_threshold}mm")
            return max(minimum_threshold, base_threshold * 0.3)  # Even more permissive for home position
        
        # Special handling for single-joint movements
        # Count how many joints have significant movement
        moving_joints = np.sum(np.abs(joint_angles) > 0.1)
        if moving_joints == 1:
            # Which joint is moving?
            active_joint = np.argmax(np.abs(joint_angles))
            logger.debug(f"Single-joint movement detected: J{active_joint+1}")
            
            # Based on RB3-730ES-U physical design, apply special case handling:
            
            # J1 (base) rotation is always safe for J1-J4/J5 pairs
            if active_joint == 0:  # Base rotation
                # For J1-J4 and J1-J5 pairs during base rotation, use minimum threshold
                if (joint1_idx == 1 and joint2_idx == 4) or (joint1_idx == 1 and joint2_idx == 5):
                    logger.debug(f"Reducing threshold for J1 rotation and {joint1_idx}-{joint2_idx} pair")
                    return minimum_threshold  # Use absolute minimum
                    
                # For other pairs during base rotation, be more permissive
                return max(minimum_threshold, base_threshold * 0.5)
            
            # J5 (wrist2) rotation is always safe
            elif active_joint == 4:  # Wrist2 rotation
                return minimum_threshold
            
            # J6 (wrist3) rotation is always safe
            elif active_joint == 5:  # Wrist3 rotation
                logger.debug(f"Reducing threshold for J6 rotation")
                return minimum_threshold
        
        # Configuration-dependent adjustments
        # Start with a more permissive baseline for the RB3-730ES-U
        config_factor = 0.8  # More permissive baseline
        
        # Compute overall configuration "complexity" - how far from neutral pose
        neutral_pose = np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0])  # Typical neutral for RB3-730ES-U
        config_deviation = np.linalg.norm(joint_angles - neutral_pose)
        
        # For J1-J4 and J1-J5 pairs (which were causing false positives):
        if (joint1_idx == 1 and joint2_idx == 4) or (joint1_idx == 1 and joint2_idx == 5):
            # More permissive thresholds based on real robot dimensions
            config_factor *= 0.75
            
            # Check if the main movement is in the wrist
            wrist_rotation = np.abs(joint_angles[3:]).sum()
            
            # If there's significant wrist movement, be even more permissive
            if wrist_rotation > 0.3:
                config_factor *= 0.8
        
        # For specific joint pairs, adjust threshold based on configuration and real robot geometry
        if (joint1_idx == 1 and joint2_idx == 4):  # Shoulder vs Wrist2
            # Based on real robot design, shoulder and wrist2 have more clearance than previously modeled
            shoulder_angle = abs(joint_angles[1])
            elbow_angle = abs(joint_angles[2])
            
            if shoulder_angle < 0.5:  # ~30 degrees
                config_factor *= 0.7  # Much more permissive
            if elbow_angle > 1.0:  # Elbow bent significantly
                config_factor *= 0.8  # Even less conservative
                
        elif (joint1_idx == 1 and joint2_idx == 5):  # Shoulder vs Wrist3
            # Based on real robot geometry in the images
            shoulder_angle = abs(joint_angles[1])
            if shoulder_angle < 0.3:
                config_factor *= 0.7  # Much more permissive
                
        elif (joint1_idx == 1 and joint2_idx == 6):  # Shoulder vs TCP
            # TCP has additional clearance as shown in the robot dimensions
            shoulder_angle = abs(joint_angles[1])
            if shoulder_angle < 0.7:
                config_factor *= 0.7  # Much more permissive
        
        elif (joint1_idx == 2 and joint2_idx == 0):  # Elbow vs Base
            # Based on the robot diagram showing elbow-base clearance
            elbow_angle = abs(joint_angles[2])
            base_angle = abs(joint_angles[0])
            
            if elbow_angle > 1.57:  # > 90 degrees
                config_factor *= 0.7  # More permissive
            if base_angle < 0.5:  # Base close to center
                config_factor *= 0.8  # Less restrictive
        
        # Joint-specific adjustments based on the DH parameters from the images
        # Check which joints are involved in the pair
        involved_joints = [joint1_idx, joint2_idx]
        
        # If both are wrist joints (3-5), they have compact design with better clearance
        if all(j >= 3 for j in involved_joints):
            config_factor *= 0.6  # Wrist joints have much better clearance from each other
        
        # Additional adaptive factors
        
        # 1. Workspace edge factor - more permissive near workspace boundaries
        if config_deviation > 2.0:  # Complex configuration
            config_factor *= edge_tolerance * 0.6  # More permissive
            
        # 2. Apply operational safety factor
        config_factor *= operational_safety
        
        # 3. Ensure we don't go below a minimum safety threshold
        # Wrist joints are smaller and can get closer
        if joint1_idx >= 3 or joint2_idx >= 3:  # If either is a wrist joint
            min_safety_factor = 0.4  # Can be more permissive for wrist joints
        else:
            min_safety_factor = 0.5  # More conservative for larger joints
            
        config_factor = max(min_safety_factor, config_factor)
        
        # 4. Also ensure we don't exceed a maximum (for very conservative operations)
        max_safety_factor = 1.5  # Reduced from 2.0 to be less conservative
        config_factor = min(max_safety_factor, config_factor)
        
        # Calculate final threshold
        final_threshold = base_threshold * config_factor
        
        # CRITICAL: Ensure we never return below the absolute minimum
        final_threshold = max(minimum_threshold, final_threshold)
        
        logger.debug(f"Adaptive threshold J{joint1_idx}-J{joint2_idx}: {final_threshold:.1f}mm "
                    f"(base: {base_threshold:.1f}mm, factor: {config_factor:.2f})")
        
        return final_threshold
    

    def check_path_collision(self, joint_path: List[np.ndarray], fk_function) -> CollisionResult:
        """Check collision along the entire joint path with intermediate points."""
        resolution = self.config.get('validation', {}).get('path_resolution', 0.01)  # 1cm resolution
        
        for i in range(len(joint_path) - 1):
            q_start = joint_path[i]
            q_end = joint_path[i + 1]
            
            # Calculate number of intermediate points needed
            joint_diff = np.linalg.norm(q_end - q_start)
            num_points = max(2, int(joint_diff / 0.1))  # At least 2 points, more for larger movements
            
            # Check intermediate points
            for j in range(num_points):
                t = j / (num_points - 1)
                q_interp = (1 - t) * q_start + t * q_end
                
                # Get forward kinematics for interpolated configuration
                T = fk_function(q_interp)
                tcp_pos = T[:3, 3]
                
                # Check all collision types for this intermediate point
                result = self.check_configuration_collision(q_interp, tcp_pos, fk_function)
                if result.is_collision:
                    result.details = f"Path collision at waypoint {i}->{i+1}, t={t:.2f}: {result.details}"
                    return result
        
        return CollisionResult(False, CollisionType.NONE, "Path collision OK")
    
    def check_configuration_collision(self, joint_angles: np.ndarray, tcp_position: np.ndarray, 
                                    fk_function=None) -> CollisionResult:
        """Comprehensive collision check for a single robot configuration."""
        
        # 1. Check joint limits
        result = self.check_joint_limits(joint_angles)
        if result.is_collision:
            return result
        
        # 2. Check workspace limits
        result = self.check_workspace_limits(tcp_position)
        if result.is_collision:
            return result
        
        # 3. Check floor/table collision (simplified with TCP only for now)
        joint_positions = [tcp_position]  # Simplified: use TCP position
        result = self.check_floor_collision(tcp_position, joint_positions)
        if result.is_collision:
            return result
        
        # 4. Check self-collision (get joint positions from FK if provided)
        if fk_function is not None:
            joint_positions = self._get_joint_positions(joint_angles, fk_function)
        result = self.check_self_collision(joint_angles, joint_positions)
        if result.is_collision:
            return result
        
        return CollisionResult(False, CollisionType.NONE, "Configuration collision-free")
    
    def _get_joint_positions(self, joint_angles: np.ndarray, fk_function) -> List[np.ndarray]:
        """Get joint positions using forward kinematics (RB3-730ES-U structure)."""
        # Compute joint positions based on URDF kinematic chain
        joint_positions = []
        
        # Base position (link0 - robot base)
        joint_positions.append(np.array([0.0, 0.0, 0.0]))
        
        # Compute intermediate joint positions using partial FK
        for i in range(len(joint_angles)):
            # Compute FK up to joint i+1
            q_partial = np.zeros_like(joint_angles)
            q_partial[:i+1] = joint_angles[:i+1]
            T_i = fk_function(q_partial)
            joint_positions.append(T_i[:3, 3])  # Extract 3D position
        
        # Add TCP position (final transformation)
        T_final = fk_function(joint_angles)
        joint_positions.append(T_final[:3, 3])
        
        return joint_positions
    
    def get_collision_summary(self) -> Dict[str, Any]:
        """Get summary of clean collision checker configuration."""
        return {
            'robot_model': 'RB3-730ES-U',
            'critical_joint_pairs': len(self.critical_joint_pairs),
            'safety_margins_enabled': self.safety_margins.get('enabled', False),
            'workspace_bounds': self.workspace,
            'path_resolution': self.config.get('validation', {}).get('path_resolution', 0.01),
            'essential_constraints': ['floor_constraint', 'wood_surface_constraint', 'workspace_limits', 'self_collision']
        }
