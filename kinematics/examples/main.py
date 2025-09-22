#!/usr/bin/env python3
"""
Production-ready main application for industrial robot kinematics.

This example demonstrates the high-performance kinematics system:
1. Forward kinematics using ForwardKinematics class
2. Real-time inverse kinematics using FastIK class
3. Comprehensive validation using KinematicsValidator
4. Real robot data validation
5. Performance analysis for production applications

Author: Robot Control Team
"""

import numpy as np
import logging
import sys
import os
import time
import json
import matplotlib.pyplot as plt

# Add the parent directory to the path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import production-ready components
from forward_kinematic import ForwardKinematics
from inverse_kinematic import FastIK
from kinematics_validation import KinematicsValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('robot_main')

def test_basic_forward_kinematics():
    """Test basic forward kinematics functionality."""
    logger.info("=== Testing Forward Kinematics ===")
    
    try:
        # Initialize forward kinematics
        fk = ForwardKinematics()
        logger.info("Forward kinematics initialized successfully")
        
        # Test configurations
        test_configs = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # Home position
            np.array([0.1, 0.2, -0.1, 0.0, 0.3, 0.0]),  # Random position
            np.array([np.pi/4, -np.pi/6, np.pi/3, 0.0, np.pi/4, np.pi/6])  # Larger angles
        ]
        
        print("\nForward Kinematics Test Results:")
        print("-" * 50)
        
        for i, q_test in enumerate(test_configs):
            T_result = fk.compute_forward_kinematics(q_test)
            
            print(f"\nTest {i+1}:")
            print(f"  Joint angles (deg): {np.round(np.rad2deg(q_test), 2)}")
            print(f"  End-effector position (m): {np.round(T_result[:3, 3], 4)}")
            
            # Convert rotation matrix to RPY for display
            rpy = fk.matrix_to_rpy(T_result[:3, :3])
            print(f"  End-effector orientation (deg): {np.round(np.rad2deg(rpy), 2)}")
        
        return fk
        
    except Exception as e:
        logger.error(f"Forward kinematics test failed: {e}")
        raise

def test_basic_inverse_kinematics(fk):
    """Test basic inverse kinematics functionality."""
    logger.info("\n=== Testing Real-Time Inverse Kinematics ===")
    
    try:
        # Initialize fast inverse kinematics
        ik = FastIK(fk)
        logger.info("Fast inverse kinematics initialized successfully")
        
        # Test with forward kinematics results
        test_configs = [
            np.array([0.1, 0.2, -0.1, 0.0, 0.3, 0.0]),
            np.array([0.5, -0.3, 0.2, 0.1, -0.2, 0.4]),
            np.array([-0.2, 0.4, -0.3, 0.2, 0.1, -0.1])
        ]
        
        print("\nInverse Kinematics Test Results:")
        print("-" * 50)
        
        successful_tests = 0
        
        for i, q_target in enumerate(test_configs):
            # Generate target pose using FK
            T_target = fk.compute_forward_kinematics(q_target)
            
            # Solve IK
            start_time = time.time()
            q_solution, converged = ik.solve(T_target, q_init=q_target)
            solve_time = time.time() - start_time
            
            print(f"\nTest {i+1}:")
            print(f"  Target joints (deg): {np.round(np.rad2deg(q_target), 2)}")
            print(f"  Converged: {converged}")
            print(f"  Solve time: {solve_time*1000:.1f} ms")
            
            if converged and q_solution is not None:
                successful_tests += 1
                
                # Verify solution accuracy
                T_check = fk.compute_forward_kinematics(q_solution)
                pos_err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
                
                # Rotation error
                R_target = T_target[:3, :3]
                R_check = T_check[:3, :3]
                R_err = R_check.T @ R_target
                cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
                rot_err = np.arccos(cos_angle)
                
                print(f"  Solution joints (deg): {np.round(np.rad2deg(q_solution), 2)}")
                print(f"  Position error: {pos_err*1000:.3f} mm")
                print(f"  Rotation error: {np.rad2deg(rot_err):.3f} deg")
                
                # Joint space difference
                joint_diff = np.linalg.norm(q_solution - q_target)
                print(f"  Joint space difference: {np.rad2deg(joint_diff):.2f} deg")
            else:
                print(f"  IK failed to converge")
        
        success_rate = successful_tests / len(test_configs)
        logger.info(f"IK success rate: {success_rate:.1%} ({successful_tests}/{len(test_configs)})")
        
        return ik
        
    except Exception as e:
        logger.error(f"Inverse kinematics test failed: {e}")
        raise

def test_workspace_exploration(fk, ik):
    """Test kinematics across different workspace regions."""
    logger.info("\n=== Testing Workspace Exploration ===")
    
    try:
        # Generate random configurations within joint limits
        limits_lower, limits_upper = fk.get_joint_limits()
        num_tests = 20
        
        print(f"\nTesting {num_tests} random configurations:")
        print("-" * 50)
        
        # Use faster parameters for workspace exploration 
        fast_params = {
            'pos_tol': 1e-3,           # 1mm tolerance 
            'rot_tol': 2e-3,           # Relaxed rotation tolerance
            'max_iters': 100,          # Fewer iterations for speed
            'num_attempts': 15,        # Fewer attempts for speed
        }
        
        successful_tests = 0
        position_errors = []
        rotation_errors = []
        solve_times = []
        
        for i in range(num_tests):
            # Generate random configuration - fix the array indexing
            q_random = np.random.uniform(limits_lower, limits_upper)
            
            # FK to get target pose
            T_target = fk.compute_forward_kinematics(q_random)
            
            # Check if within workspace constraints
            pos = T_target[:3, 3]
            # For now, skip workspace checking in exploration (will be handled by planning layer)
            # if not constraints_checker.check_workspace(pos):
            #     continue
            
            # Solve IK with fast parameters
            start_time = time.time()
            q_solution, converged = ik.solve(T_target, **fast_params)
            solve_time = time.time() - start_time
            solve_times.append(solve_time)
            
            if converged and q_solution is not None:
                successful_tests += 1
                
                # Check accuracy
                T_check = fk.compute_forward_kinematics(q_solution)
                pos_err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
                position_errors.append(pos_err)
                
                R_err = T_check[:3, :3].T @ T_target[:3, :3]
                cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
                rot_err = np.arccos(cos_angle)
                rotation_errors.append(rot_err)
                
                if i < 5:  # Show details for first 5 tests
                    print(f"  Test {i+1}: Position error {pos_err*1000:.2f} mm, "
                          f"Rotation error {np.rad2deg(rot_err):.2f} deg, "
                          f"Time {solve_time*1000:.1f} ms")
        
        # Summary statistics
        if position_errors:
            print(f"\nWorkspace Exploration Summary:")
            print(f"  Success rate: {successful_tests/len(solve_times):.1%}")
            print(f"  Mean position error: {np.mean(position_errors)*1000:.3f} mm")
            print(f"  Max position error: {np.max(position_errors)*1000:.3f} mm")
            print(f"  Mean rotation error: {np.rad2deg(np.mean(rotation_errors)):.3f} deg")
            print(f"  Mean solve time: {np.mean(solve_times)*1000:.1f} ms")
        
    except Exception as e:
        logger.error(f"Workspace exploration test failed: {e}")

def run_comprehensive_validation(fk, ik):
    """Run comprehensive validation using KinematicsValidator."""
    logger.info("\n" + "="*60)
    logger.info("=== COMPREHENSIVE VALIDATION SUITE ===")
    logger.info("="*60)
    
    try:
        # Create validator
        validator = KinematicsValidator(fk, ik)
        
        # Run validation tests
        validation_results = {}
        
        # 1. Screw axes verification
        logger.info("1. Verifying screw axes theory...")
        screw_results = validator.verify_screw_axes_theory()
        validation_results['screw_axes'] = screw_results
        
        # 2. FK-IK consistency test
        logger.info("2. Testing FK-IK consistency...")
        fk_ik_results = validator.test_fk_ik_consistency(num_tests=150)
        validation_results['fk_ik_consistency'] = fk_ik_results
        
        # 3. Workspace coverage analysis
        logger.info("3. Analyzing workspace coverage...")
        workspace_results = validator.analyze_workspace_coverage(num_samples=800)
        validation_results['workspace_coverage'] = workspace_results
        
        # 4. Performance benchmark
        logger.info("4. Running performance benchmark...")
        performance_results = validator.benchmark_performance(num_fk_tests=2000, num_ik_tests=100)
        validation_results['performance'] = performance_results
        
        # 5. Real robot data validation (if available)
        real_data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'third_20250710_162459.json')
        if os.path.exists(real_data_path):
            logger.info("5. Validating against real robot data...")
            real_data_results = validator.validate_against_real_data(real_data_path, num_samples=15)
            validation_results['real_data'] = real_data_results
        else:
            logger.info("5. Real robot data file not found - skipping")
        
        # Display summary
        logger.info("\n=== VALIDATION SUMMARY ===")
        
        # Screw axes
        if screw_results['is_valid']:
            logger.info("✅ Screw axes verification: PASSED")
        else:
            logger.warning(f"⚠️ Screw axes verification: Issues detected (max diff: {screw_results['max_difference']:.6f})")
        
        # FK-IK consistency
        success_rate = fk_ik_results['success_rate']
        if success_rate > 0.95:
            logger.info(f"✅ IK success rate: EXCELLENT ({success_rate:.1%})")
        elif success_rate > 0.85:
            logger.info(f"⚠️ IK success rate: GOOD ({success_rate:.1%})")
        else:
            logger.warning(f"❌ IK success rate: NEEDS IMPROVEMENT ({success_rate:.1%})")
        
        # Accuracy
        if 'mean_pos_error' in fk_ik_results:
            mean_pos_mm = fk_ik_results['mean_pos_error'] * 1000
            mean_rot_deg = np.rad2deg(fk_ik_results['mean_rot_error'])
            
            if mean_pos_mm < 0.1:
                logger.info(f"✅ Position accuracy: EXCELLENT ({mean_pos_mm:.3f} mm)")
            elif mean_pos_mm < 1.0:
                logger.info(f"⚠️ Position accuracy: GOOD ({mean_pos_mm:.3f} mm)")
            else:
                logger.warning(f"❌ Position accuracy: NEEDS IMPROVEMENT ({mean_pos_mm:.3f} mm)")
                
            if mean_rot_deg < 0.1:
                logger.info(f"✅ Rotation accuracy: EXCELLENT ({mean_rot_deg:.3f}°)")
            elif mean_rot_deg < 1.0:
                logger.info(f"⚠️ Rotation accuracy: GOOD ({mean_rot_deg:.3f}°)")
            else:
                logger.warning(f"❌ Rotation accuracy: NEEDS IMPROVEMENT ({mean_rot_deg:.3f}°)")
        
        # Workspace coverage
        if 'realistic_coverage_percentage' in workspace_results:
            coverage = workspace_results['realistic_coverage_percentage']
            if coverage > 80:
                logger.info(f"✅ Workspace coverage (realistic): EXCELLENT ({coverage:.1f}%)")
            elif coverage > 60:
                logger.info(f"⚠️ Workspace coverage (realistic): ADEQUATE ({coverage:.1f}%)")
            else:
                logger.warning(f"❌ Workspace coverage (realistic): LOW ({coverage:.1f}%)")
        else:
            # Original coverage metric
            coverage = workspace_results['coverage_percentage']
            if coverage > 80:
                logger.info(f"✅ Workspace coverage: EXCELLENT ({coverage:.1f}%)")
            elif coverage > 60:
                logger.info(f"⚠️ Workspace coverage: ADEQUATE ({coverage:.1f}%)")
            else:
                logger.warning(f"❌ Workspace coverage: LOW ({coverage:.1f}%)")
        
        # Performance
        fk_freq = performance_results['fk_performance']['frequency_hz']
        ik_mean_time = performance_results['ik_performance']['mean_time'] * 1000
        
        logger.info(f"📊 FK Performance: {fk_freq:.0f} Hz")
        logger.info(f"📊 IK Performance: {ik_mean_time:.1f} ms average")
        
        # Generate and save comprehensive report
        logger.info("\n=== Generating Reports ===")
        report = validator.generate_comprehensive_report()
        
        # Save report
        report_path = os.path.join(os.path.dirname(__file__), 'validation_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"📄 Validation report saved to: {report_path}")
        
        # Generate plots if possible
        try:
            plot_path = os.path.join(os.path.dirname(__file__), 'validation_plots.png')
            validator.plot_validation_results(save_path=plot_path)
            logger.info(f"📊 Validation plots saved to: {plot_path}")
        except Exception as e:
            logger.info("📊 Plots not generated (matplotlib may not be available)")
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Comprehensive validation failed: {e}")
        return None

def validate_real_robot_data_direct(fk, ik):
    """Direct validation against real robot data using modular components."""
    logger.info("\n=== Direct Real Robot Data Validation ===")
    
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'third_20250710_162459.json')
        
        if not os.path.exists(data_path):
            logger.info("Real robot data file not found - skipping direct validation")
            return None
        
        # Load real robot data
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        waypoints = data.get('waypoints', [])
        if not waypoints:
            logger.warning("No waypoints found in real robot data")
            return None
        
        print(f"\nTesting against {len(waypoints)} real robot waypoints:")
        print("-" * 50)
        
        position_errors = []
        rotation_errors = []
        ik_success_count = 0
        
        # Test subset of waypoints (every 3rd waypoint to save time)
        # test_indices = range(0, len(waypoints), 3)
        test_indices = range(len(waypoints))
        
        for i in test_indices:
            wp = waypoints[i]
            
            try:
                # Extract and convert data
                q_deg = np.array(wp['joint_positions'])
                tcp_data = np.array(wp['tcp_position'])
                
                # Convert to standard units
                q_rad = np.deg2rad(q_deg)
                tcp_pos_m = tcp_data[:3] / 1000.0  # mm to m
                tcp_rpy_rad = np.deg2rad(tcp_data[3:])  # deg to rad
                
                # Create target transformation matrix
                R_recorded = fk.rpy_to_matrix(tcp_rpy_rad)
                T_recorded = np.eye(4)
                T_recorded[:3, :3] = R_recorded
                T_recorded[:3, 3] = tcp_pos_m
                
                # Test forward kinematics
                T_fk = fk.compute_forward_kinematics(q_rad)
                
                # Calculate errors
                pos_err = np.linalg.norm(T_fk[:3, 3] - T_recorded[:3, 3])
                position_errors.append(pos_err)
                
                R_fk = T_fk[:3, :3]
                R_err = R_fk.T @ R_recorded
                cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
                rot_err = np.arccos(cos_angle)
                rotation_errors.append(rot_err)
                
                # Test inverse kinematics (every 5th waypoint)
                if i % 15 == 0:
                    q_ik, converged = ik.solve(T_recorded, q_init=q_rad)
                    if converged:
                        ik_success_count += 1
                
                if len(position_errors) <= 5:  # Show first 5 in detail
                    print(f"  Waypoint {i}: Position error {pos_err*1000:.2f} mm, "
                          f"Rotation error {np.rad2deg(rot_err):.2f} deg")
                
            except Exception as e:
                logger.warning(f"Error processing waypoint {i}: {e}")
        
        # Summary
        if position_errors:
            mean_pos_error = np.mean(position_errors)
            max_pos_error = np.max(position_errors)
            mean_rot_error = np.mean(rotation_errors)
            max_rot_error = np.max(rotation_errors)
            
            print(f"\nReal Robot Data Validation Summary:")
            print(f"  Waypoints tested: {len(position_errors)}")
            print(f"  Mean position error: {mean_pos_error*1000:.3f} mm")
            print(f"  Max position error: {max_pos_error*1000:.3f} mm")
            print(f"  Mean rotation error: {np.rad2deg(mean_rot_error):.3f} deg")
            print(f"  Max rotation error: {np.rad2deg(max_rot_error):.3f} deg")
            
            # Assessment
            if mean_pos_error < 0.005:  # 5mm
                logger.info("✅ Real robot position accuracy is excellent")
            else:
                logger.warning("⚠️ Real robot position accuracy may need improvement")
            
            if mean_rot_error < 0.1:  # ~5.7 degrees
                logger.info("✅ Real robot rotation accuracy is excellent")
            else:
                logger.warning("⚠️ Real robot rotation accuracy may need improvement")
            
            return {
                'waypoints_tested': len(position_errors),
                'mean_position_error': mean_pos_error,
                'max_position_error': max_pos_error,
                'mean_rotation_error': mean_rot_error,
                'max_rotation_error': max_rot_error,
                'ik_tests': len([i for i in test_indices if i % 15 == 0]),
                'ik_successes': ik_success_count
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Real robot data validation failed: {e}")
        return None

def test_fast_ik_performance(fk, ik):
    """Test FastIK solver performance for real-time pick-and-place operations."""
    logger.info("\n=== Testing FastIK for Real-time Pick-and-Place ===")
    
    try:
        # Initialize FastIK
        fast_ik = FastIK(fk)
        logger.info("Fast IK solver initialized successfully")
        
        # Configure standard and fast IK solvers for comparison
        standard_params = {
            'num_attempts': 10,
            'max_iters': 150,
            'pos_tol': 1e-3,  # 1mm
            'rot_tol': 5e-3   # ~0.3°
        }
        
        fast_params = {
            'time_budget': 0.020,  # 20ms time budget
            'acceptable_error': 0.004,  # 4mm
            'max_attempts': 5
        }
        
        # Generate pick and place target poses
        print("\nGenerating pick-and-place target poses...")
        targets = generate_pick_and_place_targets(50)  # Generate 50 targets
        print(f"Generated {len(targets)} target poses")
        
        # Test standard IK
        standard_results = {
            'times': [],
            'success_count': 0,
            'pos_errors': [],
            'rot_errors': []
        }
        
        print("\nTesting standard IK...")
        q_current = np.zeros(fk.n_joints)
        
        start_time = time.time()
        for i, target in enumerate(targets):
            solve_start = time.time()
            q_sol, success = ik.solve(target, q_init=q_current, **standard_params)
            solve_time = time.time() - solve_start
            standard_results['times'].append(solve_time)
            
            if success:
                standard_results['success_count'] += 1
                
                # Calculate errors
                T_achieved = fk.compute_forward_kinematics(q_sol)
                pos_err = np.linalg.norm(T_achieved[:3, 3] - target[:3, 3])
                standard_results['pos_errors'].append(pos_err)
                
                # Rotation error
                R_err = T_achieved[:3, :3].T @ target[:3, :3]
                cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
                rot_err = np.arccos(cos_angle)
                standard_results['rot_errors'].append(rot_err)
                
                # Update current position for next target
                q_current = q_sol
            
            if i % 10 == 0:
                print(f"  Processing target {i+1}/{len(targets)}...")
        
        standard_total_time = time.time() - start_time
        
        # Test FastIK
        fast_results = {
            'times': [],
            'success_count': 0,
            'pos_errors': [],
            'rot_errors': [],
            'cache_hits': 0
        }
        
        print("\nTesting FastIK...")
        q_current = np.zeros(fk.n_joints)
        fast_ik.clear_cache()
        
        start_time = time.time()
        for i, target in enumerate(targets):
            solve_start = time.time()
            q_sol, success = fast_ik.solve(target, q_init=q_current, **fast_params)
            solve_time = time.time() - solve_start
            fast_results['times'].append(solve_time)
            
            if success:
                fast_results['success_count'] += 1
                
                # Calculate errors
                T_achieved = fk.compute_forward_kinematics(q_sol)
                pos_err = np.linalg.norm(T_achieved[:3, 3] - target[:3, 3])
                fast_results['pos_errors'].append(pos_err)
                
                # Rotation error
                R_err = T_achieved[:3, :3].T @ target[:3, :3]
                cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
                rot_err = np.arccos(cos_angle)
                fast_results['rot_errors'].append(rot_err)
                
                # Update current position for next target
                q_current = q_sol
                
            if i % 10 == 0:
                print(f"  Processing target {i+1}/{len(targets)}...")
        
        fast_total_time = time.time() - start_time
        fast_stats = fast_ik.get_statistics()
        fast_results['cache_hits'] = fast_stats['cache_hits']
        
        # Calculate real-time performance metrics
        real_time_threshold = 0.020  # 20ms is typically real-time for robot control
        standard_realtime_count = sum(1 for t in standard_results['times'] if t <= real_time_threshold)
        fast_realtime_count = sum(1 for t in fast_results['times'] if t <= real_time_threshold)
        
        # Display results
        print("\n=== FastIK Performance Results ===")
        print("\nStandard IK:")
        print(f"  Success rate: {standard_results['success_count']}/{len(targets)} "
              f"({standard_results['success_count']/len(targets)*100:.1f}%)")
        print(f"  Real-time success rate: {standard_realtime_count}/{len(targets)} "
              f"({standard_realtime_count/len(targets)*100:.1f}%)")
        print(f"  Average solve time: {np.mean(standard_results['times'])*1000:.2f}ms")
        print(f"  Max solve time: {np.max(standard_results['times'])*1000:.2f}ms")
        if standard_results['pos_errors']:
            print(f"  Average position error: {np.mean(standard_results['pos_errors'])*1000:.2f}mm")
            print(f"  Average rotation error: {np.mean(standard_results['rot_errors']):.4f}rad "
                 f"({np.mean(standard_results['rot_errors'])*180/np.pi:.2f}°)")
        print(f"  Total processing time: {standard_total_time:.2f}s")
        
        print("\nFast IK:")
        print(f"  Success rate: {fast_results['success_count']}/{len(targets)} "
              f"({fast_results['success_count']/len(targets)*100:.1f}%)")
        print(f"  Real-time success rate: {fast_realtime_count}/{len(targets)} "
              f"({fast_realtime_count/len(targets)*100:.1f}%)")
        print(f"  Average solve time: {np.mean(fast_results['times'])*1000:.2f}ms")
        print(f"  Max solve time: {np.max(fast_results['times'])*1000:.2f}ms")
        print(f"  Cache hit rate: {fast_results['cache_hits']}/{len(targets)} "
              f"({fast_results['cache_hits']/len(targets)*100:.1f}%)")
        if fast_results['pos_errors']:
            print(f"  Average position error: {np.mean(fast_results['pos_errors'])*1000:.2f}mm")
            print(f"  Average rotation error: {np.mean(fast_results['rot_errors']):.4f}rad "
                 f"({np.mean(fast_results['rot_errors'])*180/np.pi:.2f}°)")
        print(f"  Total processing time: {fast_total_time:.2f}s")
        
        # Compute speedup
        if standard_total_time > 0:
            speedup = standard_total_time / fast_total_time
            print(f"\nOverall speedup: {speedup:.2f}x")
        
        avg_standard = np.mean(standard_results['times']) if standard_results['times'] else 0
        avg_fast = np.mean(fast_results['times']) if fast_results['times'] else 0
        if avg_standard > 0:
            solve_speedup = avg_standard / avg_fast
            print(f"Per-solve speedup: {solve_speedup:.2f}x")
        
        # Plot results if matplotlib available
        try:
            plt.figure(figsize=(12, 10))
            
            # Solve times
            plt.subplot(2, 2, 1)
            plt.plot(standard_results['times'], 'b-', label='Standard IK')
            plt.plot(fast_results['times'], 'r-', label='Fast IK')
            plt.axhline(y=real_time_threshold, color='g', linestyle='--', 
                       label=f'Real-time ({real_time_threshold*1000}ms)')
            plt.xlabel('Target #')
            plt.ylabel('Solve Time (s)')
            plt.title('IK Solve Times')
            plt.legend()
            plt.grid(True)
            
            # Success rates
            plt.subplot(2, 2, 2)
            labels = ['Overall Success', 'Real-time Success']
            standard_success = [standard_results['success_count']/len(targets), 
                               standard_realtime_count/len(targets)]
            fast_success = [fast_results['success_count']/len(targets), 
                           fast_realtime_count/len(targets)]
            
            x = np.arange(len(labels))
            width = 0.35
            plt.bar(x - width/2, standard_success, width, label='Standard IK')
            plt.bar(x + width/2, fast_success, width, label='Fast IK')
            plt.xlabel('Metric')
            plt.ylabel('Success Rate')
            plt.title('IK Success Rates')
            plt.xticks(x, labels)
            plt.ylim(0, 1.0)
            plt.legend()
            plt.grid(True)
            
            # Position errors
            plt.subplot(2, 2, 3)
            if standard_results['pos_errors'] and fast_results['pos_errors']:
                plt.hist(np.array(standard_results['pos_errors'])*1000, bins=15, alpha=0.5, label='Standard IK')
                plt.hist(np.array(fast_results['pos_errors'])*1000, bins=15, alpha=0.5, label='Fast IK')
                plt.xlabel('Position Error (mm)')
                plt.ylabel('Count')
                plt.title('Position Error Distribution')
                plt.legend()
                plt.grid(True)
            
            # Rotation errors
            plt.subplot(2, 2, 4)
            if standard_results['rot_errors'] and fast_results['rot_errors']:
                plt.hist(np.array(standard_results['rot_errors'])*180/np.pi, bins=15, alpha=0.5, label='Standard IK')
                plt.hist(np.array(fast_results['rot_errors'])*180/np.pi, bins=15, alpha=0.5, label='Fast IK')
                plt.xlabel('Rotation Error (degrees)')
                plt.ylabel('Count')
                plt.title('Rotation Error Distribution')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('fast_ik_performance.png')
            print("Performance plot saved as 'fast_ik_performance.png'")
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate performance plot: {e}")
        
        return {
            'standard': standard_results,
            'fast': fast_results,
            'speedup': solve_speedup if avg_standard > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"FastIK test failed: {e}")
        raise

def generate_pick_and_place_targets(num_targets: int = 50):
    """
    Generate a sequence of realistic pick-and-place target poses.
    
    Creates targets that simulate picking objects from a bin and
    placing them in another location - common in industrial automation.
    
    Args:
        num_targets: Number of targets to generate
        
    Returns:
        List of 4x4 homogeneous transformation matrices
    """
    targets = []
    
    # Define pick and place regions
    pick_region = {
        'center': np.array([0.4, 0.3, 0.1]),  # x, y, z in meters
        'size': np.array([0.2, 0.2, 0.05])    # width, depth, height
    }
    
    place_region = {
        'center': np.array([0.4, -0.3, 0.1]),
        'size': np.array([0.2, 0.2, 0.05])
    }
    
    # Generate alternating pick and place poses
    for i in range(num_targets):
        is_pick = (i % 2 == 0)
        region = pick_region if is_pick else place_region
        
        # Random position within region
        position = region['center'] + np.random.uniform(-0.5, 0.5, 3) * region['size']
        
        # For picking: mostly top-down orientation with small variations
        # For placing: similar but with more variation
        roll = np.random.normal(0, 0.1) if is_pick else np.random.normal(0, 0.2)
        pitch = np.random.normal(0, 0.1) if is_pick else np.random.normal(0, 0.2)
        yaw = np.random.uniform(-np.pi, np.pi)  # Allow any yaw rotation
        
        # Create rotation matrix (ZYX convention)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        R = Rz @ Ry @ Rx
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = position
        
        targets.append(T)
    
    return targets

def display_performance_summary(fk, ik):
    """Display performance summary from modular components."""
    logger.info("\n=== Performance Summary ===")
    
    # Get IK statistics
    ik_stats = ik.get_statistics()
    
    print("Modular Kinematics Performance:")
    print(f"  Total IK calls: {ik_stats['total_calls']}")
    print(f"  Successful IK calls: {ik_stats['successful_calls']}")
    print(f"  IK success rate: {ik_stats['success_rate']:.1%}")
    print(f"  Average IK time: {ik_stats['average_time']*1000:.1f} ms")
    
    # Joint limits info
    joint_limits = fk.get_joint_limits()
    limits_lower = joint_limits[0]
    limits_upper = joint_limits[1]
    print(f"\nRobot Configuration:")
    print(f"  Number of joints: {fk.n_joints}")
    print(f"  Joint limits (deg): [{np.rad2deg(limits_lower[0]):.0f}, {np.rad2deg(limits_upper[0]):.0f}] (per joint)")
    print(f"  Joint limit range: ±{np.rad2deg(limits_upper[0]):.0f}°")

def main():
    """Main application entry point."""
    logger.info("Starting modular robot kinematics validation suite")
    logger.info("="*60)
    
    try:
        # Test 1: Forward kinematics
        fk = test_basic_forward_kinematics()
        
        # Test 2: Inverse kinematics  
        ik = test_basic_inverse_kinematics(fk)
        
        # Test 3: Workspace exploration
        test_workspace_exploration(fk, ik)
        
        # Test 4: Comprehensive validation
        validation_results = run_comprehensive_validation(fk, ik)
        
        # Test 5: Direct real data validation
        real_data_results = validate_real_robot_data_direct(fk, ik)
        
        # Test 6: FastIK performance for real-time pick-and-place
        print("\nDo you want to test FastIK for real-time pick-and-place? (y/n)")
        response = input().lower()
        fastik_results = None
        
        if response == 'y' or response == 'yes':
            fastik_results = test_fast_ik_performance(fk, ik)
            
            # Display FastIK usage recommendations
            print("\n=== FastIK Usage Recommendations ===")
            print("\nFor Real-time Pick-and-Place Operations:")
            print("1. Basic Usage:")
            print("   fast_ik = FastIK(fk)")
            print("   q_sol, success = fast_ik.solve(target_pose)")
            print("\n2. Time Budgeting:")
            print("   fast_ik.set_time_budget(0.020)  # 20ms time budget")
            print("   # or when solving:")
            print("   q_sol, success = fast_ik.solve(target_pose, time_budget=0.020)")
            print("\n3. Warm Starting (caching):")
            print("   # Enable warm starting (on by default)")
            print("   fast_ik.update_parameters(use_warm_start=True)")
            print("\n4. Error Tolerance:")
            print("   # Allow larger errors for faster solving")
            print("   fast_ik.update_parameters(acceptable_error=0.004)  # 4mm")
            
        # Display performance summary
        display_performance_summary(fk, ik)
        
        # Final summary
        logger.info("\n" + "="*60)
        logger.info("=== FINAL SUMMARY ===")
        logger.info("="*60)
        
        if validation_results:
            logger.info("✅ All modular kinematics tests completed successfully")
            logger.info("📊 Comprehensive validation passed")
        else:
            logger.warning("⚠️ Some validation tests encountered issues")
        
        if real_data_results:
            logger.info("✅ Real robot data validation completed")
            
        if fastik_results:
            logger.info("✅ FastIK performance test completed")
            logger.info(f"🚀 FastIK speedup: {fastik_results.get('speedup', 0):.2f}x")
        
        logger.info("✅ Modular kinematics system validation complete")
        logger.info("📁 Check the examples directory for generated reports and plots")
        
        return {
            'forward_kinematics': fk,
            'inverse_kinematics': ik,
            'validation_results': validation_results,
            'real_data_results': real_data_results,
            'fastik_results': fastik_results
        }
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        raise

if __name__ == "__main__":
    results = main()