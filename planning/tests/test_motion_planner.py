#!/usr/bin/env python3
"""
Unit Tests for Motion Planning Module

Comprehensive test suite covering:
- MotionPlanner initialization and configuration
- Thread safety for concurrent operations
- Planning strategies (joint space, Cartesian, hybrid)
- IK solving with constraints
- Error handling and validation
- Statistics tracking

Author: Robot Control Team
"""

import sys
import os
import unittest
import numpy as np
import threading
import time
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import modules to test
from planning.src.motion_planner import (
    MotionPlanner, PlanningStrategy, PlanningStatus, 
    MotionPlan, MotionPlanningResult
)


class MockCollisionChecker:
    """Mock collision checker for testing."""
    
    def check_configuration_collision(self, q, tcp_position, fk_func):
        """Mock collision check - always returns no collision."""
        result = Mock()
        result.is_collision = False
        result.details = ""
        return result
    
    def check_path_collision(self, joint_path, fk_func):
        """Mock path collision check - always returns no collision."""
        result = Mock()
        result.is_collision = False
        result.details = ""
        return result
    
    def get_collision_summary(self):
        """Mock collision summary."""
        return {"total_checks": 0, "collisions_detected": 0}


class MockForwardKinematics:
    """Mock forward kinematics for testing."""
    
    def __init__(self):
        self.tool_attached = False
        
    def compute_forward_kinematics(self, q):
        """Mock FK computation."""
        # Return a reasonable 4x4 transformation matrix
        T = np.eye(4)
        T[0, 3] = q[0] * 0.1  # Simple mock position
        T[1, 3] = q[1] * 0.1
        T[2, 3] = 0.8 + q[2] * 0.1
        return T
    
    def compute_tcp_kinematics(self, q):
        """Mock TCP computation."""
        return self.compute_forward_kinematics(q)
    
    def is_tool_attached(self):
        return self.tool_attached
    
    def attach_tool(self, tool_name):
        self.tool_attached = True
        return True
    
    def detach_tool(self):
        self.tool_attached = False


class MockInverseKinematics:
    """Mock inverse kinematics for testing."""
    
    def solve(self, target_pose, q_init=None):
        """Mock IK solver."""
        # Simple mock: return a reasonable joint configuration
        if q_init is None:
            q_init = np.zeros(6)
        
        # Add some noise to simulate solution
        q_solution = q_init + np.random.normal(0, 0.1, 6)
        q_solution = np.clip(q_solution, -np.pi, np.pi)
        
        # Mock convergence (90% success rate)
        converged = np.random.random() > 0.1
        return q_solution, converged


class MockPathPlanner:
    """Mock path planner for testing."""
    
    def __init__(self, fk, ik):
        self.fk = fk
        self.ik = ik
        
    def plan_path(self, start_config, goal_config, **kwargs):
        """Mock path planning."""
        result = Mock()
        result.success = True
        result.path = [start_config, goal_config]  # Simple straight line
        result.error_message = ""
        result.validation_results = {}
        return result
    
    def validate_joint_path(self, waypoints):
        """Mock path validation."""
        result = Mock()
        result.success = True
        result.validation_results = {}
        result.error_message = ""
        return result
    
    def get_joint_limits(self):
        """Mock joint limits."""
        return {
            'j1': {'min': -np.pi, 'max': np.pi},
            'j2': {'min': -np.pi, 'max': np.pi},
            'j3': {'min': -np.pi, 'max': np.pi},
            'j4': {'min': -np.pi, 'max': np.pi},
            'j5': {'min': -np.pi, 'max': np.pi},
            'j6': {'min': -np.pi, 'max': np.pi},
        }


class MockTrajectoryPlanner:
    """Mock trajectory planner for testing."""
    
    def __init__(self, path_planner):
        self.path_planner = path_planner
        
    def plan_trajectory(self, waypoints, **kwargs):
        """Mock trajectory planning."""
        result = Mock()
        result.success = True
        result.trajectory = Mock()
        result.error_message = ""
        return result


class TestMotionPlanner(unittest.TestCase):
    """Test cases for MotionPlanner class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_fk = MockForwardKinematics()
        self.mock_ik = MockInverseKinematics()
        self.mock_path_planner = MockPathPlanner(self.mock_fk, self.mock_ik)
        self.mock_trajectory_planner = MockTrajectoryPlanner(self.mock_path_planner)
        self.mock_collision_checker = MockCollisionChecker()
        
        # Mock the collision checker import and initialization
        with patch('planning.src.motion_planner.EnhancedCollisionChecker', return_value=self.mock_collision_checker):
            self.planner = MotionPlanner(
                kinematics_fk=self.mock_fk,
                kinematics_ik=self.mock_ik,
                path_planner=self.mock_path_planner,
                trajectory_planner=self.mock_trajectory_planner
            )
    
    def test_initialization(self):
        """Test motion planner initialization."""
        self.assertIsNotNone(self.planner)
        self.assertEqual(self.planner.fk, self.mock_fk)
        self.assertEqual(self.planner.ik, self.mock_ik)
        self.assertIsNotNone(self.planner._planning_lock)
        self.assertIsNotNone(self.planner._stats_lock)
        self.assertIsNotNone(self.planner._config_lock)
        
        # Check default configuration
        self.assertEqual(self.planner.config['default_strategy'], PlanningStrategy.JOINT_SPACE)
        self.assertEqual(self.planner.config['max_planning_time'], 30.0)
        self.assertEqual(self.planner.config['default_waypoint_count'], 10)
    
    def test_thread_safety_locks(self):
        """Test that thread safety locks are properly initialized."""
        self.assertIsInstance(self.planner._planning_lock, type(threading.RLock()))
        self.assertIsInstance(self.planner._stats_lock, type(threading.RLock()))
        self.assertIsInstance(self.planner._config_lock, type(threading.RLock()))
    
    def test_configuration_updates(self):
        """Test thread-safe configuration updates."""
        new_config = {'max_planning_time': 60.0, 'default_waypoint_count': 20}
        self.planner.update_config(new_config)
        
        self.assertEqual(self.planner.config['max_planning_time'], 60.0)
        self.assertEqual(self.planner.config['default_waypoint_count'], 20)
    
    def test_statistics_tracking(self):
        """Test statistics tracking with thread safety."""
        initial_stats = self.planner.get_statistics()
        self.assertEqual(initial_stats['total_plans'], 0)
        self.assertEqual(initial_stats['successful_plans'], 0)
        self.assertEqual(initial_stats['failed_plans'], 0)
        
        # Mock successful result
        result = MotionPlanningResult(status=PlanningStatus.SUCCESS)
        self.planner._update_statistics(result)
        
        stats = self.planner.get_statistics()
        self.assertEqual(stats['successful_plans'], 1)
        
        # Mock failed result
        result = MotionPlanningResult(status=PlanningStatus.FAILED)
        self.planner._update_statistics(result)
        
        stats = self.planner.get_statistics()
        self.assertEqual(stats['failed_plans'], 1)
    
    def test_joint_space_planning(self):
        """Test joint space motion planning."""
        start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal_config = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        result = self.planner.plan_motion(
            start_config, goal_config,
            strategy=PlanningStrategy.JOINT_SPACE
        )
        
        self.assertEqual(result.status, PlanningStatus.SUCCESS)
        self.assertIsNotNone(result.plan)
        self.assertGreater(result.planning_time, 0)
        self.assertEqual(result.plan.strategy_used, PlanningStrategy.JOINT_SPACE)
    
    def test_cartesian_space_planning(self):
        """Test Cartesian space motion planning."""
        start_pose = np.eye(4)
        start_pose[:3, 3] = [0.5, 0.0, 0.8]
        
        goal_pose = np.eye(4)
        goal_pose[:3, 3] = [0.6, 0.1, 0.9]
        
        result = self.planner.plan_cartesian_motion(start_pose, goal_pose)
        
        self.assertIn(result.status, [PlanningStatus.SUCCESS, PlanningStatus.IK_FAILED])
        if result.status == PlanningStatus.SUCCESS:
            self.assertIsNotNone(result.plan)
            self.assertIsNotNone(result.plan.cartesian_waypoints)
    
    def test_hybrid_planning_strategy(self):
        """Test hybrid planning strategy."""
        start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal_config = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        result = self.planner.plan_motion(
            start_config, goal_config,
            strategy=PlanningStrategy.HYBRID
        )
        
        # Should succeed with either joint space or Cartesian space
        self.assertEqual(result.status, PlanningStatus.SUCCESS)
        self.assertEqual(result.plan.strategy_used, PlanningStrategy.HYBRID)
    
    def test_ik_solving_with_constraints(self):
        """Test constrained IK solving."""
        target_pose = np.eye(4)
        target_pose[:3, 3] = [0.5, 0.0, 0.8]
        
        # Mock collision checker to always return no collision
        with patch.object(self.planner, '_validate_ik_solution', return_value=True):
            q_solution, success = self.planner.solve_constrained_ik(target_pose)
            
            if success:
                self.assertIsNotNone(q_solution)
                self.assertEqual(len(q_solution), 6)
    
    def test_waypoint_motion_planning(self):
        """Test motion planning through waypoints."""
        waypoints = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        ]
        
        result = self.planner.plan_waypoint_motion(waypoints)
        
        self.assertEqual(result.status, PlanningStatus.SUCCESS)
        self.assertIsNotNone(result.plan)
        self.assertGreaterEqual(len(result.plan.joint_waypoints), 3)
    
    def test_invalid_waypoint_count(self):
        """Test planning with insufficient waypoints."""
        waypoints = [np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
        
        result = self.planner.plan_waypoint_motion(waypoints)
        
        self.assertEqual(result.status, PlanningStatus.FAILED)
        self.assertIn("at least 2 waypoints", result.error_message)
    
    def test_progress_callback(self):
        """Test progress callback functionality."""
        progress_updates = []
        
        def mock_callback(progress, message):
            progress_updates.append((progress, message))
        
        self.planner.set_progress_callback(mock_callback)
        self.planner.enable_progress_feedback(True)
        
        start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal_config = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        
        result = self.planner.plan_motion(start_config, goal_config)
        
        # Should have received progress updates
        self.assertGreater(len(progress_updates), 0)
        
        # Check for expected progress values
        progress_values = [update[0] for update in progress_updates]
        self.assertIn(10.0, progress_values)  # Validation
        self.assertIn(100.0, progress_values)  # Completion
    
    def test_error_handling(self):
        """Test error handling in planning operations."""
        # Test with invalid configuration (None) - should trigger constraint violation
        result = self.planner.plan_motion(None, None)
        
        # The validation catches None inputs as constraint violations
        self.assertEqual(result.status, PlanningStatus.CONSTRAINT_VIOLATION)
        self.assertIsNotNone(result.error_message)
    
    def test_fallback_strategies(self):
        """Test fallback strategy mechanism."""
        # Mock primary strategy to fail
        with patch.object(self.planner, '_plan_joint_space') as mock_joint:
            mock_joint.return_value = MotionPlanningResult(
                status=PlanningStatus.FAILED,
                error_message="Primary strategy failed"
            )
            
            start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            goal_config = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
            
            result = self.planner.plan_motion(
                start_config, goal_config,
                strategy=PlanningStrategy.JOINT_SPACE
            )
            
            # Should attempt fallback strategies
            self.assertGreater(result.attempts_made, 1)


class TestThreadSafety(unittest.TestCase):
    """Test cases for thread safety."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_fk = MockForwardKinematics()
        self.mock_ik = MockInverseKinematics()
        self.mock_path_planner = MockPathPlanner(self.mock_fk, self.mock_ik)
        self.mock_trajectory_planner = MockTrajectoryPlanner(self.mock_path_planner)
        self.mock_collision_checker = MockCollisionChecker()
        
        with patch('planning.src.motion_planner.EnhancedCollisionChecker', return_value=self.mock_collision_checker):
            self.planner = MotionPlanner(
                kinematics_fk=self.mock_fk,
                kinematics_ik=self.mock_ik,
                path_planner=self.mock_path_planner,
                trajectory_planner=self.mock_trajectory_planner
            )
    
    def test_concurrent_planning(self):
        """Test concurrent planning operations."""
        def planning_task(task_id):
            """Individual planning task."""
            start_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            goal_config = np.array([0.1 * task_id, 0.1, 0.1, 0.1, 0.1, 0.1])
            
            result = self.planner.plan_motion(start_config, goal_config)
            return result.status == PlanningStatus.SUCCESS
        
        # Run multiple planning tasks concurrently
        num_tasks = 5
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            futures = [executor.submit(planning_task, i) for i in range(num_tasks)]
            results = [future.result() for future in as_completed(futures)]
        
        # All tasks should succeed
        self.assertEqual(sum(results), num_tasks)
        
        # Statistics should be correct
        stats = self.planner.get_statistics()
        self.assertEqual(stats['total_plans'], num_tasks)
    
    def test_concurrent_config_updates(self):
        """Test concurrent configuration updates."""
        def config_update_task(task_id):
            """Individual config update task."""
            config = {'test_param_' + str(task_id): task_id * 10}
            self.planner.update_config(config)
            return True
        
        # Run multiple config updates concurrently
        num_tasks = 10
        with ThreadPoolExecutor(max_workers=num_tasks) as executor:
            futures = [executor.submit(config_update_task, i) for i in range(num_tasks)]
            results = [future.result() for future in as_completed(futures)]
        
        # All updates should succeed
        self.assertEqual(sum(results), num_tasks)
        
        # Check that all config parameters were added
        for i in range(num_tasks):
            param_name = 'test_param_' + str(i)
            self.assertIn(param_name, self.planner.config)
            self.assertEqual(self.planner.config[param_name], i * 10)
    
    def test_statistics_thread_safety(self):
        """Test statistics updates under concurrent access."""
        def stats_update_task():
            """Task that triggers statistics update."""
            result = MotionPlanningResult(status=PlanningStatus.SUCCESS)
            self.planner._update_statistics(result)
            return True
        
        # Run multiple statistics updates concurrently
        num_tasks = 20
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(stats_update_task) for _ in range(num_tasks)]
            results = [future.result() for future in as_completed(futures)]
        
        # All updates should succeed
        self.assertEqual(sum(results), num_tasks)
        
        # Final count should be correct
        stats = self.planner.get_statistics()
        self.assertEqual(stats['successful_plans'], num_tasks)


class TestValidation(unittest.TestCase):
    """Test cases for validation and error handling."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_fk = MockForwardKinematics()
        self.mock_ik = MockInverseKinematics()
        self.mock_path_planner = MockPathPlanner(self.mock_fk, self.mock_ik)
        self.mock_trajectory_planner = MockTrajectoryPlanner(self.mock_path_planner)
        self.mock_collision_checker = MockCollisionChecker()
        
        with patch('planning.src.motion_planner.EnhancedCollisionChecker', return_value=self.mock_collision_checker):
            self.planner = MotionPlanner(
                kinematics_fk=self.mock_fk,
                kinematics_ik=self.mock_ik,
                path_planner=self.mock_path_planner,
                trajectory_planner=self.mock_trajectory_planner
            )
    
    def test_joint_limit_validation(self):
        """Test joint limit validation."""
        # Test configuration outside joint limits
        invalid_config = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Beyond ±π
        valid_config = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        validation = self.planner._validate_joint_config(invalid_config)
        self.assertFalse(validation['valid'])
        self.assertIn("outside limits", validation['error'])
        
        validation = self.planner._validate_joint_config(valid_config)
        self.assertTrue(validation['valid'])
    
    def test_ik_solution_validation(self):
        """Test IK solution validation."""
        target_pose = np.eye(4)
        target_pose[:3, 3] = [0.5, 0.0, 0.8]
        
        # Test with valid solution - use a configuration that matches the mock FK output
        q_solution = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Will give position [0.5, 0.0, 0.8]
        
        # The mock collision checker already returns no collision
        is_valid = self.planner._validate_ik_solution(q_solution, target_pose)
        self.assertTrue(is_valid)
    
    def test_motion_path_validation(self):
        """Test motion path validation."""
        joint_path = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
            np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        ]
        
        # Mock collision checker already returns no collisions
        is_valid, message = self.planner.validate_motion_path(joint_path)
        self.assertTrue(is_valid)
        self.assertEqual(message, "Path validation successful")


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestMotionPlanner))
    suite.addTest(unittest.makeSuite(TestThreadSafety))
    suite.addTest(unittest.makeSuite(TestValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%")
    
    if result.failures:
        print(f"\nFAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)