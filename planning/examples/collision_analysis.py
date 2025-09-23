#!/usr/bin/env python3
"""
Collision Detection Analysis

Analyze the collision detection system to understand why safe configurations
are being flagged as collisions and assess the robustness of the system.

Author: Robot Control Team
"""

import sys
import os
import numpy as np
import logging

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'kinematics', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_collision_detection():
    """Analyze collision detection system."""
    logger.info("COLLISION DETECTION ANALYSIS")
    logger.info("=" * 50)
    
    # Import modules
    from forward_kinematic import ForwardKinematics
    from collision_checker import EnhancedCollisionChecker
    
    fk = ForwardKinematics()
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'constraints.yaml')
    collision_checker = EnhancedCollisionChecker(config_path)
    
    logger.info("Collision detection system initialized")
    
    # Test configurations that should be safe
    test_configs = [
        ("Home position", np.zeros(6)),
        ("Small positive movements", np.radians([5, 5, 5, 5, 5, 5])),
        ("Small negative movements", np.radians([-5, -5, -5, -5, -5, -5])),
        ("Only J1 rotation", np.radians([30, 0, 0, 0, 0, 0])),
        ("Only J6 rotation", np.radians([0, 0, 0, 0, 0, 30])),
        ("J2-J3 movements", np.radians([0, -30, 30, 0, 0, 0])),
        ("Typical working pose", np.radians([0, -30, 60, 0, 30, 0])),
    ]
    
    collision_results = []
    
    for name, q in test_configs:
        logger.info(f"\nTesting: {name}")
        logger.info(f"   Joint angles: {np.rad2deg(q).round(1)}°")
        
        # Compute forward kinematics
        T_tcp = fk.compute_forward_kinematics(q)
        tcp_position = T_tcp[:3, 3]
        
        logger.info(f"   TCP position: {(tcp_position*1000).round(1)}mm")
        
        # Check each collision type individually
        
        # 1. Joint limits
        joint_result = collision_checker.check_joint_limits(q)
        
        # 2. Workspace limits  
        workspace_result = collision_checker.check_workspace_limits(tcp_position)
        
        # 3. Floor collision
        joint_positions = [tcp_position]  # Simplified
        floor_result = collision_checker.check_floor_collision(tcp_position, joint_positions)
        
        # 4. Self-collision (this is likely the problem)
        joint_positions_full = collision_checker._get_joint_positions(q, fk.compute_forward_kinematics)
        self_collision_result = collision_checker.check_self_collision(q, joint_positions_full)
        
        # 5. Overall check
        overall_result = collision_checker.check_configuration_collision(q, tcp_position, fk.compute_forward_kinematics)
        
        # Report results
        logger.info(f"   Joint limits: {'OK' if not joint_result.is_collision else 'FAILED - ' + joint_result.details}")
        logger.info(f"   Workspace: {'OK' if not workspace_result.is_collision else 'FAILED - ' + workspace_result.details}")
        logger.info(f"   Floor/surface: {'OK' if not floor_result.is_collision else 'FAILED - ' + floor_result.details}")
        logger.info(f"   Self-collision: {'OK' if not self_collision_result.is_collision else 'FAILED - ' + self_collision_result.details}")
        logger.info(f"   Overall: {'OK' if not overall_result.is_collision else 'FAILED - ' + overall_result.details}")
        
        collision_results.append({
            'name': name,
            'config': q,
            'tcp_position': tcp_position,
            'joint_limits_ok': not joint_result.is_collision,
            'workspace_ok': not workspace_result.is_collision,
            'floor_ok': not floor_result.is_collision,
            'self_collision_ok': not self_collision_result.is_collision,
            'overall_ok': not overall_result.is_collision,
            'self_collision_details': self_collision_result.details if self_collision_result.is_collision else None
        })
    
    # Analysis summary
    logger.info(f"\nCOLLISION ANALYSIS SUMMARY")
    logger.info("=" * 50)
    
    total_configs = len(test_configs)
    joint_limit_violations = sum(1 for r in collision_results if not r['joint_limits_ok'])
    workspace_violations = sum(1 for r in collision_results if not r['workspace_ok'])
    floor_violations = sum(1 for r in collision_results if not r['floor_ok'])
    self_collision_violations = sum(1 for r in collision_results if not r['self_collision_ok'])
    overall_violations = sum(1 for r in collision_results if not r['overall_ok'])
    
    logger.info(f"Total configurations tested: {total_configs}")
    logger.info(f"Joint limit violations: {joint_limit_violations}/{total_configs}")
    logger.info(f"Workspace violations: {workspace_violations}/{total_configs}")
    logger.info(f"Floor/surface violations: {floor_violations}/{total_configs}")
    logger.info(f"Self-collision violations: {self_collision_violations}/{total_configs}")
    logger.info(f"Overall violations: {overall_violations}/{total_configs}")
    
    # Identify the main issue
    if self_collision_violations > 0:
        logger.info(f"\nMAIN ISSUE: Self-collision detection")
        logger.info("Self-collision violations detected:")
        for result in collision_results:
            if not result['self_collision_ok']:
                logger.info(f"   - {result['name']}: {result['self_collision_details']}")
    
    # Check collision checker configuration
    logger.info(f"\nCOLLISION CHECKER CONFIGURATION")
    logger.info("=" * 50)
    
    logger.info(f"Critical joint pairs: {len(collision_checker.critical_joint_pairs)}")
    for pair in collision_checker.critical_joint_pairs:
        min_dist = collision_checker.min_joint_distances.get(pair, "Unknown")
        logger.info(f"   J{pair[0]}-J{pair[1]}: min distance {min_dist}mm")
    
    logger.info(f"Safety margins enabled: {collision_checker.safety_margins.get('enabled', False)}")
    logger.info(f"Workspace bounds: x=[{collision_checker.workspace.get('x_min', 'Unknown')}, {collision_checker.workspace.get('x_max', 'Unknown')}]")
    logger.info(f"                  y=[{collision_checker.workspace.get('y_min', 'Unknown')}, {collision_checker.workspace.get('y_max', 'Unknown')}]")
    logger.info(f"                  z=[{collision_checker.workspace.get('z_min', 'Unknown')}, {collision_checker.workspace.get('z_max', 'Unknown')}]")
    
    # Test adaptive thresholds
    logger.info(f"\nADAPTIVE THRESHOLD ANALYSIS")
    logger.info("=" * 50)
    
    for name, q in test_configs[:3]:  # Test first 3 configs
        logger.info(f"\n{name}:")
        for pair in collision_checker.critical_joint_pairs[:3]:  # Test first 3 pairs
            threshold = collision_checker._get_adaptive_threshold(pair[0], pair[1], q)
            base_threshold = collision_checker.min_joint_distances.get(pair, 80)
            logger.info(f"   J{pair[0]}-J{pair[1]}: {threshold:.1f}mm (base: {base_threshold}mm)")
    
    # Recommendations
    logger.info(f"\nRECOMMENDATIONS")
    logger.info("=" * 50)
    
    if self_collision_violations > 0:
        logger.info("1. Self-collision thresholds may be too conservative")
        logger.info("2. Consider adjusting minimum distances for RB3-730ES-U geometry")
        logger.info("3. Validate URDF-derived collision model against real robot")
        logger.info("4. Review adaptive threshold calculations")
    
    if overall_violations == 0:
        logger.info("Collision detection system is working correctly")
        logger.info("   All test configurations passed collision checking")
    
    logger.info("\nCOLLISION ANALYSIS COMPLETED")
    return collision_results

if __name__ == "__main__":
    analyze_collision_detection()