#!/usr/bin/env python3
"""
Thread Safety Demo for Motion Planning

This demo showcases the thread-safe motion planning capabilities
by running multiple planning operations concurrently.

Author: Robot Control Team
"""

import sys
import os
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from kinematics.src.forward_kinematic import ForwardKinematics
from kinematics.src.inverse_kinematic import InverseKinematics
from planning.src.motion_planner import MotionPlanner, PlanningStrategy

def create_motion_planner():
    """Create a motion planner instance."""
    # Initialize kinematics
    urdf_path = os.path.join(os.path.dirname(__file__), '..', '..', 'rb3_730es_u.urdf')
    fk = ForwardKinematics(urdf_path)
    ik = InverseKinematics(fk)
    
    # Create motion planner
    planner = MotionPlanner(fk, ik)
    return planner

def planning_task(planner, task_id):
    """Individual planning task."""
    print(f"🚀 Task {task_id}: Starting planning...")
    
    # Generate different start and goal configurations for each task
    start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_config = np.array([
        0.1 * task_id,
        0.1 * (task_id % 2),
        0.1 * ((task_id + 1) % 3),
        0.1 * (task_id % 4),
        0.1 * ((task_id + 2) % 2),
        0.1 * (task_id % 3)
    ])
    
    start_time = time.time()
    
    try:
        # Plan motion
        result = planner.plan_motion(
            start_config, goal_config,
            strategy=PlanningStrategy.JOINT_SPACE,
            waypoint_count=5
        )
        
        planning_time = time.time() - start_time
        
        if result.status.value == "success":
            print(f"✅ Task {task_id}: Success! {len(result.plan.joint_waypoints)} waypoints in {planning_time:.3f}s")
            return {
                'task_id': task_id,
                'success': True,
                'planning_time': planning_time,
                'waypoint_count': len(result.plan.joint_waypoints),
                'strategy': result.plan.strategy_used.value
            }
        else:
            print(f"❌ Task {task_id}: Failed - {result.error_message}")
            return {
                'task_id': task_id,
                'success': False,
                'planning_time': planning_time,
                'error': result.error_message
            }
            
    except Exception as e:
        planning_time = time.time() - start_time
        print(f"💥 Task {task_id}: Exception - {str(e)}")
        return {
            'task_id': task_id,
            'success': False,
            'planning_time': planning_time,
            'error': str(e)
        }

def test_thread_safety():
    """Test thread safety with concurrent planning operations."""
    print("🧵 THREAD SAFETY DEMO FOR MOTION PLANNING")
    print("=" * 60)
    
    # Create motion planner
    print("📋 Initializing motion planner...")
    planner = create_motion_planner()
    print("✅ Motion planner ready")
    
    # Test concurrent planning
    num_tasks = 8
    max_workers = 4
    
    print(f"\n🎯 Running {num_tasks} concurrent planning tasks with {max_workers} workers...")
    print("-" * 60)
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(planning_task, planner, task_id) 
            for task_id in range(1, num_tasks + 1)
        ]
        
        # Collect results as they complete
        for future in as_completed(futures):
            results.append(future.result())
    
    total_time = time.time() - start_time
    
    # Analyze results
    print(f"\n📊 RESULTS SUMMARY")
    print("=" * 60)
    
    successful_tasks = [r for r in results if r['success']]
    failed_tasks = [r for r in results if not r['success']]
    
    print(f"Total tasks: {len(results)}")
    print(f"Successful: {len(successful_tasks)} ({len(successful_tasks)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failed_tasks)} ({len(failed_tasks)/len(results)*100:.1f}%)")
    print(f"Total execution time: {total_time:.3f}s")
    
    if successful_tasks:
        avg_planning_time = sum(r['planning_time'] for r in successful_tasks) / len(successful_tasks)
        total_waypoints = sum(r['waypoint_count'] for r in successful_tasks)
        print(f"Average planning time: {avg_planning_time:.3f}s")
        print(f"Total waypoints generated: {total_waypoints}")
    
    # Show individual results
    print(f"\n📋 DETAILED RESULTS")
    print("-" * 60)
    for result in sorted(results, key=lambda x: x['task_id']):
        if result['success']:
            print(f"Task {result['task_id']:2d}: ✅ {result['waypoint_count']} waypoints in {result['planning_time']:.3f}s")
        else:
            print(f"Task {result['task_id']:2d}: ❌ {result['error'][:50]}...")
    
    # Check statistics
    stats = planner.get_statistics()
    print(f"\n📈 PLANNER STATISTICS")
    print("-" * 60)
    print(f"Total plans executed: {stats['total_plans']}")
    print(f"Successful plans: {stats['successful_plans']}")
    print(f"Failed plans: {stats['failed_plans']}")
    
    # Verify thread safety
    expected_total = len(results)
    actual_total = stats['total_plans']
    
    if expected_total == actual_total:
        print(f"✅ Thread safety verified: Statistics are consistent!")
    else:
        print(f"❌ Thread safety issue: Expected {expected_total}, got {actual_total}")
    
    print(f"\n🎉 Thread safety demo completed!")
    
    return len(successful_tasks) == num_tasks

if __name__ == '__main__':
    success = test_thread_safety()
    sys.exit(0 if success else 1)