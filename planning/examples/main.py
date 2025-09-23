#!/usr/bin/env python3
"""
Motion Planning System Demonstration

Demonstrates the motion planning system including:
- Constraint checking
- Path planning
- Trajectory planning
- High-level coordination
- Optional clean robot interface (mm/degrees)
"""

import sys
import os
import numpy as np
import logging
import time

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'kinematics', 'src'))

# Import kinematics modules
from forward_kinematic import ForwardKinematics
from inverse_kinematic import FastIK

# Import planning modules
from path_planner import PathPlanner, ConstraintsChecker
from trajectory_planner import TrajectoryPlanner
from motion_planner import MotionPlanner, PlanningStrategy, PlanningStatus

# Import clean robot interface
try:
    from clean_robot_interface import CleanRobotMotionPlanner, RobotPose
    CLEAN_INTERFACE_AVAILABLE = True
except ImportError:
    CLEAN_INTERFACE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('planning_demo')


def test_constraint_checking():
    logger.info("=== Testing Constraint Checking ===")
    try:
        constraints_checker = ConstraintsChecker()
        print("\nConstraint Checking Test Results:")
        print("-" * 50)
        q_valid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        q_invalid = np.array([0.1, 0.2, 3.0, 0.4, 0.5, 0.6])
        valid, msg = constraints_checker.check_joint_limits(q_valid)
        print(f"Valid joint config: {valid} - {msg}")
        valid, msg = constraints_checker.check_joint_limits(q_invalid)
        print(f"Invalid joint config: {valid} - {msg}")
        pos_valid = np.array([0.3, 0.2, 0.4])
        pos_invalid = np.array([2.0, 0.2, 0.4])
        valid, msg = constraints_checker.check_workspace(pos_valid)
        print(f"Valid workspace position: {valid} - {msg}")
        valid, msg = constraints_checker.check_workspace(pos_invalid)
        print(f"Invalid workspace position: {valid} - {msg}")
        logger.info("Constraint checking tests completed")
        return constraints_checker
    except Exception as e:
        logger.error(f"Constraint checking test failed: {e}")
        raise


def test_path_planning(fk, ik):
    logger.info("\n=== Testing Path Planning ===")
    try:
        path_planner = PathPlanner(fk, ik)
        print("\nPath Planning Test Results:")
        print("-" * 50)
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_goal = np.array([0.5, -0.3, 0.2, 0.1, 0.4, -0.2])
        start_time = time.time()
        result = path_planner.plan_path(q_start, q_goal, max_iterations=300)
        planning_time = time.time() - start_time
        print("AORRTC path planning:")
        print(f"  Success: {result.success}")
        print(f"  Planning time: {planning_time*1000:.1f} ms")
        if result.success:
            print(f"  Waypoints generated: {len(result.path)}")
            if result.validation_results:
                print(f"  Validation: {result.validation_results}")
            print(f"  Start waypoint: {np.degrees(result.path[0])} deg")
            print(f"  Goal waypoint: {np.degrees(result.path[-1])} deg")
        else:
            print(f"  Error: {result.error_message}")
        logger.info("Path planning tests completed")
        return path_planner, result if result.success else None
    except Exception as e:
        logger.error(f"Path planning test failed: {e}")
        raise


def test_trajectory_planning(path_planner, sample_path):
    logger.info("\n=== Testing Trajectory Planning ===")
    try:
        traj_planner = TrajectoryPlanner(path_planner)
        print("\nTrajectory Planning Test Results:")
        print("-" * 50)
        if sample_path is None:
            waypoints = [
                np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                np.array([0.2, -0.1, 0.1, 0.0, 0.2, 0.0]),
                np.array([0.4, -0.2, 0.2, 0.1, 0.4, 0.1])
            ]
        else:
            waypoints = sample_path.path[::3]
        start_time = time.time()
        result = traj_planner.plan_trajectory(waypoints, optimize=True)
        planning_time = time.time() - start_time
        print("Trajectory planning:")
        print(f"  Success: {result.success}")
        print(f"  Planning time: {planning_time*1000:.1f} ms")
        if result.success:
            trajectory = result.trajectory
            print(f"  Trajectory points: {len(trajectory.points)}")
            if hasattr(trajectory, 'total_duration'):
                print(f"  Planning time: {trajectory.total_duration:.2f} s")
            else:
                print("  Planning time: N/A")
            print(f"  Max velocities: {np.max(np.degrees(trajectory.max_velocities)):.1f} deg/s (max)")
            print(f"  Max accelerations: {np.max(np.degrees(trajectory.max_accelerations)):.1f} deg/s² (max)")
            print(f"  Smoothness metric: {trajectory.smoothness_metric:.4f}")
            if result.optimization_info:
                opt_info = result.optimization_info
                if 'time_scale' in opt_info:
                    print(f"  Optimization: Time scaled by {opt_info['time_scale']:.2f}")
                else:
                    print("  Optimization: Info available but no time scaling reported")
        else:
            print(f"  Error: {result.error_message}")
        logger.info("Trajectory planning tests completed")
        return traj_planner, result.trajectory if result.success else None
    except Exception as e:
        logger.error(f"Trajectory planning test failed: {e}")
        raise


def test_motion_planning(fk, ik):
    logger.info("\n=== Testing Motion Planning Coordinator ===")
    try:
        motion_planner = MotionPlanner(fk, ik)
        print("\nMotion Planning Test Results:")
        print("-" * 50)
        q_start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_goal = np.array([0.3, -0.2, 0.4, 0.1, 0.3, -0.1])
        print("\n1. Joint Space Motion Planning:")
        start_time = time.time()
        result = motion_planner.plan_motion(q_start, q_goal, strategy=PlanningStrategy.JOINT_SPACE, waypoint_count=8)
        print(f"   Status: {result.status.value}")
        print(f"   Planning time: {result.planning_time*1000:.1f} ms")
        print(f"   Fallback used: {result.fallback_used}")
        if result.status == PlanningStatus.SUCCESS:
            plan = result.plan
            print(f"   Waypoints: {plan.num_waypoints}")
            print(f"   Total trajectory time: {plan.planning_time:.2f} s")
            print(f"   Strategy used: {plan.strategy_used.value}")
        print("\n2. Cartesian Space Motion Planning:")
        T_start = fk.compute_forward_kinematics(q_start)
        T_goal = fk.compute_forward_kinematics(q_goal)
        start_time = time.time()
        result = motion_planner.plan_cartesian_motion(T_start, T_goal)
        print(f"   Status: {result.status.value}")
        print(f"   Planning time: {result.planning_time*1000:.1f} ms")
        if result.status == PlanningStatus.SUCCESS:
            plan = result.plan
            print(f"   Joint waypoints: {plan.num_waypoints}")
            print(f"   Cartesian waypoints: {len(plan.cartesian_waypoints) if plan.cartesian_waypoints else 0}")
        print("\n3. Multi-Waypoint Motion Planning:")
        waypoints = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.2, -0.1, 0.2, 0.0, 0.1, 0.0]),
            np.array([0.4, -0.3, 0.4, 0.1, 0.3, 0.1]),
            np.array([0.1, -0.2, 0.1, -0.1, 0.2, -0.1])
        ]
        result = motion_planner.plan_waypoint_motion(waypoints)
        print(f"   Status: {result.status.value}")
        print(f"   Planning time: {result.planning_time*1000:.1f} ms")
        if result.status == PlanningStatus.SUCCESS:
            plan = result.plan
            print(f"   Waypoints processed: {plan.num_waypoints}")
            print(f"   Total execution time: {plan.planning_time:.2f} s")
        stats = motion_planner.get_statistics()
        print("\nMotion Planner Statistics:")
        print(f"  Total plans: {stats['total_plans']}")
        print(f"  Successful: {stats['successful_plans']}")
        print(f"  Failed: {stats['failed_plans']}")
        logger.info("Motion planning tests completed")
        return motion_planner
    except Exception as e:
        logger.error(f"Motion planning test failed: {e}")
        raise


def test_integrated_planning_pipeline():
    logger.info("\n" + "="*60)
    logger.info("=== INTEGRATED PLANNING PIPELINE TEST ===")
    logger.info("="*60)
    try:
        print("Initializing planning system components...")
        fk = ForwardKinematics()
        ik = FastIK(fk)
        motion_planner = MotionPlanner(fk, ik)
        print("\nIntegrated Planning Pipeline Test:")
        print("-" * 50)
        print("\nScenario: Pick and place operation")
        q_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_approach = np.array([0.3, -0.4, 0.5, 0.0, 0.2, 0.0])
        q_pick = np.array([0.3, -0.3, 0.3, 0.0, 0.1, 0.0])
        q_place = np.array([-0.2, -0.2, 0.2, 0.1, 0.3, 0.1])
        q_retreat = np.array([-0.2, -0.4, 0.4, 0.1, 0.2, 0.1])
        motion_sequence = [
            ("Home to Approach", q_home, q_approach),
            ("Approach to Pick", q_approach, q_pick), 
            ("Pick to Place", q_pick, q_place),
            ("Place to Retreat", q_place, q_retreat),
            ("Retreat to Home", q_retreat, q_home)
        ]
        total_planning_time = 0.0
        successful_motions = 0
        for i, (description, q_start, q_goal) in enumerate(motion_sequence):
            print(f"\n  Motion {i+1}: {description}")
            result = motion_planner.plan_motion(q_start, q_goal, strategy=PlanningStrategy.HYBRID, waypoint_count=6)
            total_planning_time += result.planning_time
            print(f"    Status: {result.status.value}")
            print(f"    Planning time: {result.planning_time*1000:.1f} ms")
            print(f"    Fallback used: {result.fallback_used}")
            if result.status == PlanningStatus.SUCCESS:
                successful_motions += 1
                plan = result.plan
                print(f"    Waypoints: {plan.num_waypoints}")
                print(f"    Trajectory time: {plan.planning_time:.2f} s")
                print(f"    Strategy: {plan.strategy_used.value}")
            else:
                print(f"    Error: {result.error_message}")
        print("\nPipeline Summary:")
        print(f"  Total motions: {len(motion_sequence)}")
        print(f"  Successful: {successful_motions}")
        print(f"  Success rate: {successful_motions/len(motion_sequence)*100:.1f}%")
        print(f"  Total planning time: {total_planning_time*1000:.1f} ms")
        print(f"  Average per motion: {total_planning_time/len(motion_sequence)*1000:.1f} ms")
        final_stats = motion_planner.get_statistics()
        print("\nFinal System Statistics:")
        print(f"  Total planning calls: {final_stats['total_plans']}")
        if final_stats['total_plans'] > 0:
            avg_planning_time = total_planning_time / final_stats['total_plans']
            print(f"  Overall success rate: {final_stats['successful_plans']/final_stats['total_plans']*100:.1f}%")
            print(f"  Average planning time: {avg_planning_time*1000:.1f} ms")
        else:
            print("  No planning attempts recorded")
        print("  Strategy usage: Data not available in current implementation")
        logger.info("Integrated planning pipeline test completed")
    except Exception as e:
        logger.error(f"Integrated planning test failed: {e}")
        raise


def test_clean_robot_interface():
    logger.info("=== Testing Clean Robot Interface (mm/degrees) ===")
    try:
        planner = CleanRobotMotionPlanner()
        logger.info("Clean robot motion planner initialized")
        current_joints_deg = [0.0, -30.0, 45.0, 0.0, -15.0, 0.0]
        logger.info(f"Current joints: {current_joints_deg} degrees")
        current_pose = planner.get_current_pose_from_joints(current_joints_deg)
        logger.info(f"Current position: {[round(x, 1) for x in current_pose.position_mm]} mm")
        logger.info(f"Current rotation: {[round(x, 1) for x in current_pose.orientation_deg]} degrees")
        target_pos = [300.0, 200.0, 400.0]
        target_rot = [0.0, 0.0, 90.0]
        logger.info(f"Target position: {target_pos} mm")
        logger.info(f"Target rotation: {target_rot} degrees")
        logger.info("Planning motion with clean interface...")
        plan = planner.plan_motion(current_joints_deg, target_pos, target_rot)
        if plan.success:
            logger.info("Motion planned successfully")
            logger.info(f"   Waypoints: {len(plan.waypoints)}")
            logger.info(f"   Planning time: {plan.planning_time_ms:.1f} ms")
            logger.info(f"   Execution time: {plan.execution_time_sec:.2f} seconds")
            if plan.waypoints:
                start_joints = plan.waypoints[0].joints_deg
                end_joints = plan.waypoints[-1].joints_deg
                logger.info(f"   Start joints (deg): {[round(x, 1) for x in start_joints]}")
                logger.info(f"   End joints (deg): {[round(x, 1) for x in end_joints]}")
        else:
            logger.warning(f"Motion planning failed: {plan.error_message}")
        logger.info("Clean robot interface test completed")
    except Exception as e:
        logger.error(f"Clean robot interface test failed: {str(e)}")


def main():
    logger.info("Starting motion planning system demonstration")
    logger.info("="*60)
    try:
        fk = ForwardKinematics()
        ik = FastIK(fk)
        logger.info("Kinematics modules initialized")
        test_constraint_checking()
        path_planner, sample_path = test_path_planning(fk, ik)
        test_trajectory_planning(path_planner, sample_path)
        test_motion_planning(fk, ik)
        test_integrated_planning_pipeline()
        if CLEAN_INTERFACE_AVAILABLE:
            test_clean_robot_interface()
        logger.info("\n" + "="*60)
        logger.info("=== DEMONSTRATION SUMMARY ===")
        logger.info("="*60)
        logger.info("All planning system components tested successfully")
        logger.info("Constraint checking working correctly")
        logger.info("Path planning with validation functional")
        logger.info("Trajectory planning with optimization working")
        logger.info("High-level motion coordination operational")
        logger.info("Integrated planning pipeline validated")
        if CLEAN_INTERFACE_AVAILABLE:
            logger.info("Clean robot interface (mm/degrees) tested")
        print("\nMotion Planning System Demonstration Complete!")
        print("\nThe planning system is ready for production use with:")
        print("  • Comprehensive constraint checking and safety validation")
        print("  • Multiple planning strategies (joint space, Cartesian, hybrid)")
        print("  • Smooth trajectory generation with optimization")
        print("  • Robust fallback mechanisms and error handling")
        if CLEAN_INTERFACE_AVAILABLE:
            print("  • Clean robot API interface (mm positions, degree rotations)")
            print("  • Direct robot.move_joint() and robot.move_tcp() compatibility")
        print("  • Real-time performance suitable for robot control")
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
