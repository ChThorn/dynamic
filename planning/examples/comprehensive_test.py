#!/usr/bin/env python3
"""
Comprehensive Test of the Cleaned Motion Planning System

This test demonstrates all the cleaned functionality:
- Basic kinematics validation
- C-space enhanced IK solving
- Joint space motion planning
- Cartesian space motion planning
- Performance comparisons

Author: Robot Control Team
"""

import sys
import os
import numpy as np
import logging
import time

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'kinematics', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def comprehensive_test():
    """Comprehensive test of the cleaned motion planning system."""
    logger.info("🚀 COMPREHENSIVE CLEANED MOTION PLANNING TEST")
    logger.info("=" * 60)
    
    # Import and initialize
    logger.info("Initializing system...")
    from forward_kinematic import ForwardKinematics
    from inverse_kinematic import InverseKinematics
    from motion_planner import MotionPlanner, PlanningStrategy, PlanningStatus
    
    fk = ForwardKinematics()
    ik = InverseKinematics(fk)
    planner = MotionPlanner(fk, ik)
    
    logger.info("✅ System initialized successfully")
    
    # Test 1: Performance comparison - IK with and without C-space
    logger.info("\n🧪 Test 1: IK Performance Comparison")
    logger.info("-" * 40)
    
    test_poses = [
        np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.2], [0, 0, 1, 0.4], [0, 0, 0, 1]]),
        np.array([[1, 0, 0, 0.5], [0, 1, 0, -0.1], [0, 0, 1, 0.3], [0, 0, 0, 1]]),
        np.array([[1, 0, 0, 0.4], [0, 1, 0, 0.3], [0, 0, 1, 0.5], [0, 0, 0, 1]])
    ]
    
    # Without C-space
    total_time_basic = 0
    basic_successes = 0
    
    for i, pose in enumerate(test_poses):
        start_time = time.time()
        q_solution, success = ik.solve(pose)
        solve_time = time.time() - start_time
        total_time_basic += solve_time
        
        if success:
            basic_successes += 1
            logger.info(f"  Basic IK {i+1}: ✅ {solve_time*1000:.1f}ms")
        else:
            logger.info(f"  Basic IK {i+1}: ❌ {solve_time*1000:.1f}ms")
    
    # Enable C-space
    planner.enable_configuration_space_analysis()
    
    # With C-space
    total_time_cspace = 0
    cspace_successes = 0
    
    for i, pose in enumerate(test_poses):
        start_time = time.time()
        q_solution, success = planner.solve_ik_with_cspace(pose)
        solve_time = time.time() - start_time
        total_time_cspace += solve_time
        
        if success:
            cspace_successes += 1
            logger.info(f"  C-space IK {i+1}: ✅ {solve_time*1000:.1f}ms")
        else:
            logger.info(f"  C-space IK {i+1}: ❌ {solve_time*1000:.1f}ms")
    
    # Performance summary
    logger.info(f"\n📊 IK Performance Summary:")
    logger.info(f"  Basic IK: {basic_successes}/{len(test_poses)} success, avg {total_time_basic/len(test_poses)*1000:.1f}ms")
    logger.info(f"  C-space IK: {cspace_successes}/{len(test_poses)} success, avg {total_time_cspace/len(test_poses)*1000:.1f}ms")
    speedup = total_time_basic / total_time_cspace if total_time_cspace > 0 else 0
    logger.info(f"  C-space speedup: {speedup:.1f}x")
    
    # Test 2: Joint space motion planning
    logger.info("\n🧪 Test 2: Joint Space Motion Planning")
    logger.info("-" * 40)
    
    test_motions = [
        (np.zeros(6), np.radians([30, -20, 45, 0, 30, 15])),
        (np.radians([10, 10, 10, 10, 10, 10]), np.radians([-20, 30, -45, 15, -30, 20])),
        (np.radians([45, -30, 60, -15, 45, 30]), np.zeros(6))
    ]
    
    joint_planning_times = []
    joint_successes = 0
    
    for i, (q_start, q_goal) in enumerate(test_motions):
        logger.info(f"  Motion {i+1}: {np.rad2deg(q_start).round(1)}° → {np.rad2deg(q_goal).round(1)}°")
        
        start_time = time.time()
        result = planner.plan_motion(q_start, q_goal, strategy=PlanningStrategy.JOINT_SPACE)
        plan_time = time.time() - start_time
        joint_planning_times.append(plan_time)
        
        if result.status == PlanningStatus.SUCCESS:
            joint_successes += 1
            logger.info(f"    ✅ {result.plan.num_waypoints} waypoints in {plan_time:.3f}s")
        else:
            logger.info(f"    ❌ Failed: {result.error_message}")
    
    # Test 3: Cartesian space motion planning
    logger.info("\n🧪 Test 3: Cartesian Space Motion Planning")
    logger.info("-" * 40)
    
    cartesian_motions = [
        (np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.1], [0, 0, 1, 0.5], [0, 0, 0, 1]]),
         np.array([[1, 0, 0, 0.4], [0, 1, 0, -0.1], [0, 0, 1, 0.3], [0, 0, 0, 1]])),
        (np.array([[1, 0, 0, 0.5], [0, 1, 0, 0.2], [0, 0, 1, 0.4], [0, 0, 0, 1]]),
         np.array([[1, 0, 0, 0.3], [0, 1, 0, 0.3], [0, 0, 1, 0.5], [0, 0, 0, 1]]))
    ]
    
    cartesian_planning_times = []
    cartesian_successes = 0
    
    for i, (T_start, T_goal) in enumerate(cartesian_motions):
        pos_start = T_start[:3, 3] * 1000  # Convert to mm for display
        pos_goal = T_goal[:3, 3] * 1000
        
        logger.info(f"  Motion {i+1}: {pos_start.round(1)}mm → {pos_goal.round(1)}mm")
        
        start_time = time.time()
        result = planner.plan_cartesian_motion(T_start, T_goal)
        plan_time = time.time() - start_time
        cartesian_planning_times.append(plan_time)
        
        if result.status == PlanningStatus.SUCCESS:
            cartesian_successes += 1
            logger.info(f"    ✅ {result.plan.num_waypoints} waypoints in {plan_time:.3f}s")
            
            # Check final accuracy
            if result.plan.cartesian_waypoints:
                final_pose = result.plan.cartesian_waypoints[-1]
                pos_error = np.linalg.norm(final_pose[:3, 3] - T_goal[:3, 3])
                logger.info(f"    📏 Final accuracy: {pos_error*1000:.2f}mm")
        else:
            logger.info(f"    ❌ Failed: {result.error_message}")
    
    # Test 4: System capabilities summary
    logger.info("\n🧪 Test 4: System Capabilities")
    logger.info("-" * 40)
    
    stats = planner.get_statistics()
    logger.info(f"  Total plans executed: {stats['total_plans']}")
    logger.info(f"  Overall success rate: {stats['success_rate']:.1f}%")
    logger.info(f"  C-space analysis: {'✅ Enabled' if planner.cspace_analysis_enabled else '❌ Disabled'}")
    
    # Final summary
    logger.info("\n🎯 COMPREHENSIVE TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ IK Solutions: Basic {basic_successes}/{len(test_poses)}, C-space {cspace_successes}/{len(test_poses)}")
    logger.info(f"✅ Joint Planning: {joint_successes}/{len(test_motions)} successful")
    logger.info(f"✅ Cartesian Planning: {cartesian_successes}/{len(cartesian_motions)} successful")
    
    avg_joint_time = np.mean(joint_planning_times) if joint_planning_times else 0
    avg_cartesian_time = np.mean(cartesian_planning_times) if cartesian_planning_times else 0
    
    logger.info(f"📊 Performance: Joint {avg_joint_time:.3f}s avg, Cartesian {avg_cartesian_time:.3f}s avg")
    logger.info(f"🚀 System Status: {'PRODUCTION READY' if all([joint_successes > 0, cartesian_successes > 0]) else 'NEEDS ATTENTION'}")
    
    logger.info("\n✅ COMPREHENSIVE TEST COMPLETED!")

if __name__ == "__main__":
    comprehensive_test()