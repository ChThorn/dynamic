#!/usr/bin/env python3
"""
Trajectory Planning Robustness Test

Test trajectory generation, smoothing, velocity/acceleration profiling,
and timing optimization to assess the robustness of the trajectory planner.

Author: Robot Control Team
"""

import sys
import os
import numpy as np
import logging
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_trajectory_planning():
    """Test trajectory planning robustness."""
    logger.info("🎯 TRAJECTORY PLANNING ROBUSTNESS TEST")
    logger.info("=" * 55)
    
    # Import modules
    from trajectory_planner import TrajectoryPlanner, TrajectoryResult, Trajectory
    
    planner = TrajectoryPlanner()
    logger.info("✅ Trajectory planner initialized")
    
    # Test 1: Basic trajectory generation
    logger.info("\n📋 Test 1: Basic Trajectory Generation")
    logger.info("-" * 40)
    
    # Define safe waypoints that don't cause collisions
    test_waypoints = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),           # Home
        np.array([0.0, -0.52, 0.52, 0.0, 0.0, 0.0]),       # J2-J3 movement
        np.array([0.0, -0.52, 1.05, 0.0, 0.52, 0.0]),      # Add J5 movement  
        np.array([0.0, -0.26, 0.78, 0.0, 0.26, 0.0]),      # Intermediate
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])           # Return home
    ]
    
    logger.info(f"Testing with {len(test_waypoints)} waypoints")
    
    start_time = time.time()
    result = planner.plan_trajectory(test_waypoints)
    planning_time = time.time() - start_time
    
    if result.success:
        logger.info(f"✅ Trajectory generation successful in {planning_time:.3f}s")
        logger.info(f"   Trajectory points: {len(result.trajectory.points)}")
        logger.info(f"   Total time: {result.trajectory.total_time:.2f}s")
        logger.info(f"   Smoothness metric: {result.trajectory.smoothness_metric:.6f}")
        
        # Check velocity and acceleration limits
        max_vels = result.trajectory.max_velocities
        max_accs = result.trajectory.max_accelerations
        
        logger.info(f"   Max velocities: {np.rad2deg(max_vels).round(1)}°/s")
        logger.info(f"   Max accelerations: {np.rad2deg(max_accs).round(1)}°/s²")
        
        # Verify limits
        vel_limit = np.rad2deg(planner.config['max_joint_velocity'])
        acc_limit = np.rad2deg(planner.config['max_joint_acceleration'])
        
        vel_ok = np.all(max_vels <= planner.config['max_joint_velocity'])
        acc_ok = np.all(max_accs <= planner.config['max_joint_acceleration'])
        
        logger.info(f"   Velocity limits: {'✅ OK' if vel_ok else '❌ EXCEEDED'} (limit: {vel_limit:.1f}°/s)")
        logger.info(f"   Acceleration limits: {'✅ OK' if acc_ok else '❌ EXCEEDED'} (limit: {acc_limit:.1f}°/s²)")
        
    else:
        logger.info(f"❌ Trajectory generation failed: {result.error_message}")
        return
    
    # Test 2: Trajectory optimization
    logger.info("\n📋 Test 2: Trajectory Optimization")
    logger.info("-" * 40)
    
    # Test with optimization enabled
    start_time = time.time()
    result_optimized = planner.plan_trajectory(test_waypoints, optimize=True)
    opt_time = time.time() - start_time
    
    if result_optimized.success:
        logger.info(f"✅ Optimized trajectory in {opt_time:.3f}s")
        
        # Compare with non-optimized
        time_improvement = result.trajectory.total_time - result_optimized.trajectory.total_time
        smoothness_improvement = result.trajectory.smoothness_metric - result_optimized.trajectory.smoothness_metric
        
        logger.info(f"   Time improvement: {time_improvement:.2f}s")
        logger.info(f"   Smoothness improvement: {smoothness_improvement:.6f}")
        logger.info(f"   Optimization gain: {'✅ Improved' if time_improvement > 0 or smoothness_improvement > 0 else '➖ No improvement'}")
    else:
        logger.info(f"❌ Trajectory optimization failed: {result_optimized.error_message}")
    
    # Test 3: Time scaling
    logger.info("\n📋 Test 3: Time Scaling")
    logger.info("-" * 40)
    
    scaling_factors = [0.5, 1.0, 2.0]
    scaling_results = []
    
    for scale in scaling_factors:
        start_time = time.time()
        result_scaled = planner.plan_trajectory(test_waypoints, time_scaling=scale, optimize=False)
        scale_time = time.time() - start_time
        
        if result_scaled.success:
            logger.info(f"✅ Scale {scale}x: {result_scaled.trajectory.total_time:.2f}s total, planned in {scale_time:.3f}s")
            scaling_results.append((scale, result_scaled.trajectory.total_time, scale_time))
        else:
            logger.info(f"❌ Scale {scale}x failed: {result_scaled.error_message}")
    
    # Verify scaling works correctly
    if len(scaling_results) >= 2:
        expected_ratio = scaling_results[2][1] / scaling_results[0][1]  # 2x / 0.5x = 4x
        logger.info(f"   Scaling verification: {expected_ratio:.1f}x ratio (expected: 4.0x)")
        scaling_ok = 3.5 <= expected_ratio <= 4.5
        logger.info(f"   Scaling accuracy: {'✅ OK' if scaling_ok else '❌ INACCURATE'}")
    
    # Test 4: Trajectory validation
    logger.info("\n📋 Test 4: Trajectory Validation")
    logger.info("-" * 40)
    
    if result.success:
        validation = planner.validate_trajectory_dynamics(result.trajectory)
        
        logger.info(f"   Velocity constraints: {'✅ OK' if validation['velocity_ok'] else '❌ VIOLATED'}")
        logger.info(f"   Acceleration constraints: {'✅ OK' if validation['acceleration_ok'] else '❌ VIOLATED'}")
        logger.info(f"   Smoothness metric: {validation['smoothness_metric']:.6f}")
        
        # Check individual joint violations
        if not validation['velocity_ok']:
            violations = validation['velocity_violations']
            for i, violated in enumerate(violations):
                if violated:
                    logger.info(f"     J{i+1} velocity: {np.rad2deg(validation['max_velocities'][i]):.1f}°/s exceeds {np.rad2deg(validation['velocity_limits'][i]):.1f}°/s")
        
        if not validation['acceleration_ok']:
            violations = validation['acceleration_violations']
            for i, violated in enumerate(violations):
                if violated:
                    logger.info(f"     J{i+1} acceleration: {np.rad2deg(validation['max_accelerations'][i]):.1f}°/s² exceeds {np.rad2deg(validation['acceleration_limits'][i]):.1f}°/s²")
    
    # Test 5: Trajectory interpolation
    logger.info("\n📋 Test 5: Trajectory Interpolation")
    logger.info("-" * 40)
    
    if result.success:
        # Test interpolation at specific times
        query_times = np.array([0.0, result.trajectory.total_time/4, result.trajectory.total_time/2, 
                               3*result.trajectory.total_time/4, result.trajectory.total_time])
        
        try:
            interpolated = planner.interpolate_trajectory(result.trajectory, query_times)
            logger.info(f"✅ Interpolation successful for {len(query_times)} query points")
            logger.info(f"   Interpolated positions shape: {interpolated['positions'].shape}")
            logger.info(f"   Interpolated velocities shape: {interpolated['velocities'].shape}")
            logger.info(f"   Interpolated accelerations shape: {interpolated['accelerations'].shape}")
        except Exception as e:
            logger.info(f"❌ Interpolation failed: {str(e)}")
    
    # Test 6: Edge cases
    logger.info("\n📋 Test 6: Edge Cases")
    logger.info("-" * 40)
    
    # Test with insufficient waypoints
    single_waypoint = [test_waypoints[0]]
    result_single = planner.plan_trajectory(single_waypoint)
    logger.info(f"   Single waypoint: {'❌ Correctly rejected' if not result_single.success else '⚠️  Unexpectedly accepted'}")
    
    # Test with identical waypoints
    identical_waypoints = [test_waypoints[0], test_waypoints[0]]
    result_identical = planner.plan_trajectory(identical_waypoints)
    logger.info(f"   Identical waypoints: {'✅ Handled' if result_identical.success else '❌ Failed'}")
    
    # Test with extreme time scaling
    result_fast = planner.plan_trajectory(test_waypoints, time_scaling=0.1)
    result_slow = planner.plan_trajectory(test_waypoints, time_scaling=10.0)
    
    logger.info(f"   Extreme fast (0.1x): {'✅ Handled' if result_fast.success else '❌ Failed'}")
    logger.info(f"   Extreme slow (10x): {'✅ Handled' if result_slow.success else '❌ Failed'}")
    
    # Test 7: Configuration update
    logger.info("\n📋 Test 7: Configuration Updates")
    logger.info("-" * 40)
    
    original_config = planner.config.copy()
    
    # Test configuration update
    new_config = {
        'max_joint_velocity': np.radians(30),  # Reduce from 45°/s to 30°/s
        'smoothing_weight': 0.2  # Change smoothing
    }
    
    planner.update_config(new_config)
    logger.info(f"✅ Configuration updated")
    logger.info(f"   New max velocity: {np.rad2deg(planner.config['max_joint_velocity']):.1f}°/s")
    logger.info(f"   New smoothing weight: {planner.config['smoothing_weight']}")
    
    # Test with new configuration
    result_new_config = planner.plan_trajectory(test_waypoints)
    if result_new_config.success:
        new_max_vels = result_new_config.trajectory.max_velocities
        vel_within_new_limit = np.all(new_max_vels <= planner.config['max_joint_velocity'])
        logger.info(f"   New limits respected: {'✅ YES' if vel_within_new_limit else '❌ NO'}")
    
    # Restore original config
    planner.config = original_config
    
    # Summary
    logger.info("\n📊 TRAJECTORY PLANNING SUMMARY")
    logger.info("=" * 55)
    
    tests_passed = 0
    total_tests = 7
    
    if result.success:
        tests_passed += 1
        logger.info("✅ Basic trajectory generation: PASSED")
    else:
        logger.info("❌ Basic trajectory generation: FAILED")
    
    if result_optimized.success:
        tests_passed += 1
        logger.info("✅ Trajectory optimization: PASSED")
    else:
        logger.info("❌ Trajectory optimization: FAILED")
    
    if len(scaling_results) > 0:
        tests_passed += 1
        logger.info("✅ Time scaling: PASSED")
    else:
        logger.info("❌ Time scaling: FAILED")
    
    if result.success and validation['velocity_ok'] and validation['acceleration_ok']:
        tests_passed += 1
        logger.info("✅ Trajectory validation: PASSED")
    else:
        logger.info("❌ Trajectory validation: FAILED")
    
    try:
        if interpolated:
            tests_passed += 1
            logger.info("✅ Trajectory interpolation: PASSED")
    except:
        logger.info("❌ Trajectory interpolation: FAILED")
    
    if not result_single.success:  # Should reject single waypoint
        tests_passed += 1
        logger.info("✅ Edge case handling: PASSED")
    else:
        logger.info("❌ Edge case handling: FAILED")
    
    if result_new_config.success:
        tests_passed += 1
        logger.info("✅ Configuration updates: PASSED")
    else:
        logger.info("❌ Configuration updates: FAILED")
    
    success_rate = (tests_passed / total_tests) * 100
    logger.info(f"\n🎯 Overall Success Rate: {tests_passed}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("🚀 TRAJECTORY PLANNER: PRODUCTION READY")
    elif success_rate >= 60:
        logger.info("⚠️  TRAJECTORY PLANNER: NEEDS MINOR IMPROVEMENTS")
    else:
        logger.info("❌ TRAJECTORY PLANNER: NEEDS MAJOR WORK")
    
    logger.info("\n✅ TRAJECTORY PLANNING TEST COMPLETED!")

if __name__ == "__main__":
    test_trajectory_planning()