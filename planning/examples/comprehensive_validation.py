#!/usr/bin/env python3
"""
Comprehensive validation test to demonstrate improved motion planning performance.

Tests the system with various scenarios to verify fixes are working correctly.
"""

import numpy as np
import time
import sys
import os

# Add project paths
planning_src = os.path.join(os.path.dirname(__file__), '..', 'src')
kinematics_src = os.path.join(os.path.dirname(__file__), '..', '..', 'kinematics', 'src')

sys.path.insert(0, planning_src)
sys.path.insert(0, kinematics_src)

from forward_kinematic import ForwardKinematics
from inverse_kinematic import InverseKinematics
from motion_planner import MotionPlanner, PlanningStrategy, PlanningStatus
from production_validator import ProductionValidator, ValidationLevel

def test_motion_planning_scenarios():
    """Test motion planning with various scenarios."""
    print("="*60)
    print("MOTION PLANNING PERFORMANCE TEST")
    print("="*60)
    
    # Initialize system
    fk = ForwardKinematics()
    ik = InverseKinematics(fk)
    planner = MotionPlanner(fk, ik)
    
    # Enable production mode
    planner.enable_production_mode(detailed_logging=True, validation_mode=True)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Basic Joint Motion',
            'start': np.zeros(6),
            'goal': np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0]),
            'strategy': PlanningStrategy.JOINT_SPACE
        },
        {
            'name': 'Complex Joint Motion',
            'start': np.array([0.5, -0.3, 0.8, 0.2, -0.4, 0.1]),
            'goal': np.array([-0.3, -0.7, 0.3, -0.1, 0.6, -0.2]),
            'strategy': PlanningStrategy.JOINT_SPACE
        },
        {
            'name': 'Hybrid Planning',
            'start': np.zeros(6),
            'goal': np.array([0.2, -0.4, 0.6, 0.1, 0.3, -0.1]),
            'strategy': PlanningStrategy.HYBRID
        },
        {
            'name': 'Home Position Test',
            'start': np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            'goal': np.zeros(6),  # Home position
            'strategy': PlanningStrategy.JOINT_SPACE
        },
        {
            'name': 'Large Motion',
            'start': np.array([-1.0, -1.0, 1.0, -0.5, 0.5, 0.0]),
            'goal': np.array([1.0, -0.2, 0.2, 0.5, -0.5, 1.0]),
            'strategy': PlanningStrategy.JOINT_SPACE
        }
    ]
    
    results = []
    total_time = 0
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nTest {i}: {scenario['name']}")
        print("-" * 40)
        
        start_time = time.time()
        result = planner.plan_motion(
            scenario['start'], 
            scenario['goal'], 
            strategy=scenario['strategy']
        )
        planning_time = time.time() - start_time
        total_time += planning_time
        
        success = result.status == PlanningStatus.SUCCESS
        
        print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"Planning Time: {planning_time:.3f}s")
        print(f"Error: {result.error_message if result.error_message else 'None'}")
        if success and result.plan:
            print(f"Waypoints: {result.plan.num_waypoints}")
            print(f"Strategy Used: {result.plan.strategy_used.value if result.plan.strategy_used else 'unknown'}")
        
        results.append({
            'name': scenario['name'],
            'success': success,
            'time': planning_time,
            'error': result.error_message
        })
    
    # Summary
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    
    successful = sum(1 for r in results if r['success'])
    success_rate = (successful / len(results)) * 100
    avg_time = total_time / len(results)
    
    print(f"Success Rate: {successful}/{len(results)} ({success_rate:.1f}%)")
    print(f"Average Planning Time: {avg_time:.3f}s")
    print(f"Total Test Time: {total_time:.3f}s")
    
    # Show detailed results
    print(f"\nDetailed Results:")
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"  {status} {r['name']}: {r['time']:.3f}s")
        if not r['success']:
            print(f"    Error: {r['error']}")
    
    # Get diagnostic report
    print(f"\nDiagnostic Report:")
    print("-" * 40)
    diagnostic = planner.get_diagnostic_report()
    print(f"Production Mode: {'✅' if diagnostic['production_mode'] else '❌'}")
    print(f"Planning Statistics: {diagnostic['planning_statistics']}")
    print(f"Success Rate: {diagnostic['success_rate']*100:.1f}%")
    print(f"Average Planning Time: {diagnostic['average_planning_time']:.3f}s")
    print(f"Consecutive Failures: {diagnostic['consecutive_failures']}")
    
    if diagnostic['failure_patterns']:
        print(f"Failure Patterns:")
        for pattern, count in diagnostic['failure_patterns'].items():
            print(f"  {pattern}: {count}")
    
    return success_rate >= 80.0  # Consider 80%+ success rate as good

def test_cartesian_planning():
    """Test Cartesian pose planning."""
    print("\n" + "="*60)
    print("CARTESIAN PLANNING TEST")
    print("="*60)
    
    # Initialize system
    fk = ForwardKinematics()
    ik = InverseKinematics(fk)
    planner = MotionPlanner(fk, ik)
    
    # Test poses
    start_pose = np.eye(4)
    start_pose[:3, 3] = [0.4, 0.0, 0.5]  # Reachable position
    
    goal_pose = np.eye(4)
    goal_pose[:3, 3] = [0.3, 0.2, 0.6]   # Another reachable position
    
    print("Testing Cartesian motion planning...")
    print(f"Start pose: {start_pose[:3, 3]}")
    print(f"Goal pose: {goal_pose[:3, 3]}")
    
    start_time = time.time()
    result = planner.plan_cartesian_motion(start_pose, goal_pose)
    planning_time = time.time() - start_time
    
    success = result.status == PlanningStatus.SUCCESS
    
    print(f"\nStatus: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"Planning Time: {planning_time:.3f}s")
    print(f"Error: {result.error_message if result.error_message else 'None'}")
    
    if success and result.plan:
        print(f"Waypoints: {result.plan.num_waypoints}")
        print(f"Strategy Used: {result.plan.strategy_used.value if result.plan.strategy_used else 'unknown'}")
    
    return success

def validate_collision_fixes():
    """Validate that collision detection fixes are working."""
    print("\n" + "="*60)
    print("COLLISION DETECTION VALIDATION")
    print("="*60)
    
    from collision_checker import EnhancedCollisionChecker
    
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'constraints.yaml')
    checker = EnhancedCollisionChecker(config_path)
    
    # Test home position (should NOT be collision)
    home_pos = np.zeros(6)
    home_tcp = np.array([0, 0, 0.8])
    
    print("Testing home position collision detection...")
    result = checker.check_configuration_collision(home_pos, home_tcp)
    
    print(f"Home position collision: {'❌ YES' if result.is_collision else '✅ NO'}")
    if result.is_collision:
        print(f"Collision details: {result.details}")
        print("❌ FAILED: Home position incorrectly flagged as collision")
        return False
    else:
        print("✅ PASSED: Home position correctly identified as collision-free")
    
    # Test adaptive thresholds
    print(f"\nTesting adaptive thresholds...")
    test_config = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    thresholds_ok = True
    for pair in checker.critical_joint_pairs[:3]:  # Test first 3 pairs
        threshold = checker._get_adaptive_threshold(pair[0], pair[1], test_config)
        print(f"Pair {pair}: {threshold*1000:.1f}mm")
        if threshold < 0.005:  # Less than 5mm is too small
            print(f"❌ Threshold too small: {threshold*1000:.1f}mm")
            thresholds_ok = False
    
    if thresholds_ok:
        print("✅ PASSED: Adaptive thresholds are reasonable")
    else:
        print("❌ FAILED: Some adaptive thresholds are too small")
    
    return not result.is_collision and thresholds_ok

def main():
    """Run comprehensive validation."""
    print("COMPREHENSIVE MOTION PLANNING VALIDATION")
    print("Testing all fixes and improvements...")
    
    # Run validation tests
    collision_ok = validate_collision_fixes()
    joint_planning_ok = test_motion_planning_scenarios()
    cartesian_ok = test_cartesian_planning()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL VALIDATION SUMMARY")
    print("="*60)
    
    tests = [
        ("Collision Detection Fixes", collision_ok),
        ("Joint Space Planning", joint_planning_ok),
        ("Cartesian Planning", cartesian_ok)
    ]
    
    passed = sum(1 for _, result in tests if result)
    
    for test_name, result in tests:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    overall_success = passed == len(tests)
    
    print(f"\nOverall Result: {passed}/{len(tests)} tests passed")
    print(f"System Status: {'✅ PRODUCTION READY' if overall_success else '❌ NEEDS ATTENTION'}")
    
    if overall_success:
        print("\n🎉 ALL FIXES SUCCESSFULLY IMPLEMENTED!")
        print("The motion planning system is now production-ready with:")
        print("  • Fixed collision detection thresholds")
        print("  • Enhanced trajectory planning")
        print("  • Improved C-space analysis")
        print("  • Production monitoring and diagnostics")
        print("  • High success rate motion planning")
    else:
        print("\n⚠️ Some issues remain - review failed tests above")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    exit(main())