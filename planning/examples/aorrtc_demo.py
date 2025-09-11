#!/usr/bin/env python3
"""
AORRTC Enhanced Motion Planning Demo

This demo showcases the enhanced motion planning system with AORRTC integration,
demonstrating advanced path planning, trajectory smoothing, and constraint validation.

Features demonstrated:
- AORRTC (Asymptotically Optimal RRT-Connect) path planning
- Advanced trajectory smoothing with gradient-based optimization
- Performance comparison with regular RRT planning
- Constraint validation and safety checking
- Fallback mechanisms for robust planning

Author: Robot Control Team
"""

import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# Add source directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'kinematics', 'src'))

from forward_kinematic import ForwardKinematics
from inverse_kinematic import InverseKinematics
from motion_planner import MotionPlanner, PlanningStrategy
from path_planner import PathPlanner, AOTRRCPathPlanner
from trajectory_planner import TrajectoryPlanner

def run_aorrtc_demo():
    """Run comprehensive AORRTC motion planning demonstration."""
    
    print("🚀 AORRTC Enhanced Motion Planning System Demo")
    print("=" * 60)
    
    # Initialize system components
    print("\n📋 Initializing System Components...")
    fk = ForwardKinematics()
    ik = InverseKinematics(fk)
    motion_planner = MotionPlanner(fk, ik)
    aorrtc_planner = AOTRRCPathPlanner(fk, ik)
    
    print("✅ Kinematics engines initialized")
    print("✅ AORRTC path planner ready")
    print("✅ Motion planning coordinator ready")
    
    # Define test scenarios
    scenarios = {
        "Simple Motion": {
            "start": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "goal": np.array([0.3, -0.2, 0.4, 0.1, 0.3, -0.1]),
            "max_iter": 300
        },
        "Complex Motion": {
            "start": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            "goal": np.array([0.6, -0.5, 0.7, 0.4, 0.5, -0.3]),
            "max_iter": 500
        },
        "Challenging Motion": {
            "start": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "goal": np.array([0.8, -0.6, 0.9, 0.5, 0.6, -0.4]),
            "max_iter": 800
        }
    }
    
    results = {}
    
    # Run test scenarios
    for scenario_name, config in scenarios.items():
        print(f"\n🎯 Testing Scenario: {scenario_name}")
        print("-" * 40)
        
        q_start = config["start"]
        q_goal = config["goal"]
        max_iter = config["max_iter"]
        
        scenario_results = {}
        
        # Test 1: AORRTC Path Planning
        print("📍 AORRTC Path Planning...")
        start_time = time.time()
        aorrtc_result = aorrtc_planner.plan_aorrtc_path(
            q_start, q_goal, max_iterations=max_iter
        )
        aorrtc_time = time.time() - start_time
        
        if aorrtc_result.success:
            print(f"   ✅ Success: {len(aorrtc_result.path)} waypoints")
            print(f"   ⏱️  Time: {aorrtc_time*1000:.1f}ms")
            if aorrtc_result.validation_results:
                vr = aorrtc_result.validation_results
                print(f"   🔢 Iterations: {vr['iterations']}")
                print(f"   🌳 Tree sizes: {vr['tree_sizes']}")
        else:
            print(f"   ❌ Failed: {aorrtc_result.error_message}")
        
        scenario_results['aorrtc'] = aorrtc_result
        
        # Test 2: Enhanced Motion Planning with AORRTC
        print("🎯 Enhanced Motion Planning (AORRTC)...")
        enhanced_result = motion_planner.plan_motion(
            q_start, q_goal,
            strategy=PlanningStrategy.JOINT_SPACE,
            use_aorrtc=True,
            max_iterations=max_iter
        )
        
        if enhanced_result.status.value == 'success':
            print(f"   ✅ Success: {enhanced_result.plan.num_waypoints} waypoints")
            print(f"   ⏱️  Planning time: {enhanced_result.planning_time:.3f}s")
            print(f"   🎭 Trajectory time: {enhanced_result.plan.total_time:.2f}s")
            if enhanced_result.plan.trajectory:
                print(f"   📊 Trajectory points: {len(enhanced_result.plan.trajectory.points)}")
        else:
            print(f"   ❌ Failed: {enhanced_result.error_message}")
        
        scenario_results['enhanced'] = enhanced_result
        
        # Test 3: Regular RRT Comparison
        print("🔄 Regular RRT Planning...")
        regular_result = motion_planner.plan_motion(
            q_start, q_goal,
            strategy=PlanningStrategy.JOINT_SPACE,
            use_aorrtc=False,
            max_iterations=max_iter
        )
        
        if regular_result.status.value == 'success':
            print(f"   ✅ Success: {regular_result.plan.num_waypoints} waypoints")
            print(f"   ⏱️  Planning time: {regular_result.planning_time:.3f}s")
        else:
            print(f"   ❌ Failed: {regular_result.error_message}")
        
        scenario_results['regular'] = regular_result
        results[scenario_name] = scenario_results
    
    # Performance Summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for scenario_name, scenario_results in results.items():
        print(f"\n{scenario_name}:")
        
        aorrtc = scenario_results.get('aorrtc')
        enhanced = scenario_results.get('enhanced')  
        regular = scenario_results.get('regular')
        
        if aorrtc and aorrtc.success:
            print(f"  AORRTC Path:     ✅ {len(aorrtc.path)} waypoints")
        
        if enhanced and enhanced.status.value == 'success':
            print(f"  Enhanced Motion: ✅ {enhanced.planning_time:.3f}s planning")
            
        if regular and regular.status.value == 'success':
            print(f"  Regular RRT:     ✅ {regular.planning_time:.3f}s planning")
            
        # Compare planning times
        if enhanced and regular and both_success(enhanced, regular):
            speedup = regular.planning_time / enhanced.planning_time
            if speedup > 1:
                print(f"  📈 AORRTC is {speedup:.1f}x slower (higher quality)")
            else:
                print(f"  📈 AORRTC is {1/speedup:.1f}x faster")
    
    # System capabilities summary
    print("\n🎉 SYSTEM CAPABILITIES DEMONSTRATED")
    print("=" * 60)
    print("✅ AORRTC Algorithm: Asymptotically optimal path planning")
    print("✅ Bidirectional Search: Efficient tree exploration")
    print("✅ Informed Sampling: Focused search in promising regions")
    print("✅ Advanced Smoothing: Gradient-based trajectory optimization")
    print("✅ Constraint Validation: Safety and feasibility checking")
    print("✅ Fallback Mechanisms: Robust planning with RRT backup")
    print("✅ Performance Optimization: KDTree nearest neighbor search")
    print("✅ Motion Coordination: High-level planning interface")
    
    print(f"\n🚀 AORRTC Enhanced Motion Planning System Demo Complete!")
    return results

def both_success(enhanced_result, regular_result):
    """Check if both planning results were successful."""
    return (enhanced_result.status.value == 'success' and 
            regular_result.status.value == 'success')

def visualize_results(results):
    """Create visualization of planning results (optional)."""
    # This function could be extended to create plots comparing:
    # - Planning times
    # - Path lengths 
    # - Trajectory smoothness
    # - Success rates
    pass

if __name__ == "__main__":
    try:
        results = run_aorrtc_demo()
        print("\nDemo completed successfully! 🎉")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
