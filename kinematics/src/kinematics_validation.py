#!/usr/bin/env python3
"""
Enhanced Kinematics Validation and Testing Utilities

This module provides comprehensive validation tools for robot kinematics including:
- Screw axes verification against theoretical values
- Forward/Inverse kinematics consistency testing
- Workspace coverage analysis
- Performance benchmarking
- Real robot data validation
- Mathematical property verification

Author: Robot Control Team
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import logging
import time
import json
import os

# Try to import matplotlib for plotting (optional)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

logger = logging.getLogger(__name__)

class KinematicsValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class KinematicsValidator:
    """Comprehensive validation utilities for robot kinematics."""
    
    def __init__(self, forward_kinematics, inverse_kinematics):
        """
        Initialize validator with kinematics modules.
        
        Args:
            forward_kinematics: ForwardKinematics instance
            inverse_kinematics: FastIK instance
        """
        self.fk = forward_kinematics
        self.ik = inverse_kinematics
        self.n_joints = forward_kinematics.n_joints
        
        # More realistic workspace bounds for RB3-730ES-U (in meters)
        # Based on the robot's actual reach characteristics
        self.workspace_bounds = {
            'x_range': [-0.75, 0.75],
            'y_range': [-0.75, 0.75], 
            'z_range': [0.05, 0.9]  # Adjusted range to match actual robot height
        }
        
        # RB3-730ES-U robot parameters for workspace analysis
        self.robot_reach = 0.730  # Max reach in meters
        self.robot_min_reach = 0.12  # Min reach in meters
        self.robot_base_height = 0.1453  # Height of first joint from base
        
        # Results storage
        self.validation_results = {}
        
        logger.info("Kinematics validator initialized with constraint-free modules")
    
    def _realistic_workspace_check(self, position: np.ndarray) -> bool:
        """
        Realistic workspace check using spherical and cylindrical constraints.
        More accurately represents the RB3-730ES-U robot's reachable workspace.
        """
        # Basic bounds check
        if not (self.workspace_bounds['x_range'][0] <= position[0] <= self.workspace_bounds['x_range'][1] and
                self.workspace_bounds['y_range'][0] <= position[1] <= self.workspace_bounds['y_range'][1] and
                self.workspace_bounds['z_range'][0] <= position[2] <= self.workspace_bounds['z_range'][1]):
            return False
        
        # Distance from robot base in XY plane
        xy_distance = np.sqrt(position[0]**2 + position[1]**2)
        
        # Spherical workspace constraint (max reach)
        distance_from_shoulder = np.sqrt(xy_distance**2 + (position[2] - self.robot_base_height)**2)
        if distance_from_shoulder > self.robot_reach:
            return False
            
        # Minimum reach constraint
        if distance_from_shoulder < self.robot_min_reach:
            return False
            
        # Additional constraints for realistic workspace
        # Avoid low positions with large xy_distance (shoulder joint limitation)
        if xy_distance > 0.6 and position[2] < 0.15:
            return False
            
        # Avoid positions directly above robot (singularity region)
        if xy_distance < 0.1 and position[2] > 0.7:
            return False
            
        return True
    
    def _simple_workspace_check(self, position: np.ndarray) -> bool:
        """Simple workspace bounds check for backward compatibility."""
        return self._realistic_workspace_check(position)
    
    def verify_screw_axes_theory(self) -> Dict[str, Any]:
        """
        Verify screw axes against theoretical computation from robot geometry.
        
        Uses the known robot geometry to compute expected screw axes and compare
        with the implemented values.
        """
        logger.info("Verifying screw axes against theoretical values")
        
        # RB3-730ES-U robot geometry (from URDF/CAD data)
        robot_joints = [
            {'name': 'base_joint', 'axis': [0,0,1], 'origin': [0,0,0.1453], 'type': 'revolute'},
            {'name': 'shoulder_joint', 'axis': [0,1,0], 'origin': [0,0,0], 'type': 'revolute'},
            {'name': 'elbow_joint', 'axis': [0,1,0], 'origin': [0,-0.00645,0.286], 'type': 'revolute'},
            {'name': 'wrist1_joint', 'axis': [0,0,1], 'origin': [0,0,0], 'type': 'revolute'},
            {'name': 'wrist2_joint', 'axis': [0,1,0], 'origin': [0,0,0.344], 'type': 'revolute'},
            {'name': 'wrist3_joint', 'axis': [0,0,1], 'origin': [0,0,0], 'type': 'revolute'}
        ]
        
        # Compute theoretical screw axes
        expected_S = np.zeros((6, 6))
        T_cumulative = np.eye(4)
        
        for i, joint in enumerate(robot_joints):
            # Transform to joint frame
            joint_origin = np.array(joint['origin'])
            T_joint = np.eye(4)
            T_joint[:3, 3] = joint_origin
            T_cumulative = T_cumulative @ T_joint
            
            # Joint axis in base frame (assuming no rotation in joint frames)
            omega = np.array(joint['axis'])
            
            # Position of joint in base frame
            p = T_cumulative[:3, 3]
            
            # Screw axis: [ω, p × ω] for revolute joints
            v = np.cross(p, omega)
            expected_S[:, i] = np.hstack([omega, v])
        
        # Compare with implemented screw axes
        implemented_S = self.fk.get_screw_axes()
        S_diff = np.abs(implemented_S - expected_S)
        max_diff = np.max(S_diff)
        
        # Analyze differences
        joint_diffs = [np.max(S_diff[:, i]) for i in range(6)]
        
        results = {
            'theoretical_S': expected_S,
            'implemented_S': implemented_S,
            'max_difference': max_diff,
            'joint_differences': joint_diffs,
            'is_valid': max_diff < 1e-6,
            'difference_matrix': S_diff,
            'rms_error': np.sqrt(np.mean(S_diff**2))
        }
        
        if results['is_valid']:
            logger.info("✅ Screw axes verification PASSED")
        else:
            logger.warning(f"⚠️ Screw axes verification shows differences (max: {max_diff:.6f})")
            
        self.validation_results['screw_axes'] = results
        return results
    
    def test_fk_ik_consistency(self, num_tests: int = 200, 
                              workspace_only: bool = True) -> Dict[str, Any]:
        """
        Test forward-inverse kinematics consistency.
        
        Args:
            num_tests: Number of test configurations
            workspace_only: Only test within reachable workspace
        """
        logger.info(f"Testing FK-IK consistency with {num_tests} configurations")
        
        position_errors = []
        rotation_errors = []
        joint_errors = []
        success_count = 0
        computation_times = []
        
        joint_limits = self.fk.get_joint_limits()
        limits_lower = joint_limits[0]  # Lower limits array
        limits_upper = joint_limits[1]  # Upper limits array
        tested = 0
        attempts = 0
        max_attempts = num_tests * 3  # Prevent infinite loop
        
        while tested < num_tests and attempts < max_attempts:
            attempts += 1
            
            # Generate random configuration
            # q_test = np.random.uniform(limits_lower[0], limits_upper[1])
            q_test = np.random.uniform(limits_lower, limits_upper, size=self.n_joints)
            
            # Check if within workspace (if required)
            T_target = self.fk.compute_forward_kinematics(q_test)
            pos = T_target[:3, 3]
            
            if workspace_only and not self._simple_workspace_check(pos):
                continue
                
            tested += 1
            
            # Solve inverse kinematics
            start_time = time.time()
            q_solution, converged = self.ik.solve(T_target, q_init=q_test)
            ik_time = time.time() - start_time
            computation_times.append(ik_time)
            
            if converged and q_solution is not None:
                success_count += 1
                
                # Verify solution accuracy
                T_check = self.fk.compute_forward_kinematics(q_solution)
                
                # Position error
                pos_err = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3])
                position_errors.append(pos_err)
                
                # Rotation error
                R_target = T_target[:3, :3]
                R_check = T_check[:3, :3]
                R_err = R_check.T @ R_target
                cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
                rot_err = np.arccos(cos_angle)
                rotation_errors.append(rot_err)
                
                # Joint space error (considering periodicity)
                joint_err = self._compute_joint_error(q_test, q_solution)
                joint_errors.append(joint_err)
        
        # Compute statistics
        success_rate = success_count / tested if tested > 0 else 0.0
        
        results = {
            'num_tests': tested,
            'num_attempts': attempts,
            'success_count': success_count,
            'success_rate': success_rate,
            'position_errors': position_errors,
            'rotation_errors': rotation_errors,
            'joint_errors': joint_errors,
            'computation_times': computation_times
        }
        
        if position_errors:
            results.update({
                'mean_pos_error': np.mean(position_errors),
                'max_pos_error': np.max(position_errors),
                'std_pos_error': np.std(position_errors),
                'mean_rot_error': np.mean(rotation_errors),
                'max_rot_error': np.max(rotation_errors),
                'std_rot_error': np.std(rotation_errors),
                'mean_joint_error': np.mean(joint_errors),
                'max_joint_error': np.max(joint_errors),
                'mean_computation_time': np.mean(computation_times),
                'max_computation_time': np.max(computation_times)
            })
        
        # Log results
        logger.info(f"FK-IK Consistency Results:")
        logger.info(f"  Success rate: {success_rate:.1%}")
        if position_errors:
            logger.info(f"  Mean position error: {results['mean_pos_error']*1000:.3f} mm")
            logger.info(f"  Mean rotation error: {np.rad2deg(results['mean_rot_error']):.3f}°")
            logger.info(f"  Mean computation time: {results['mean_computation_time']*1000:.1f} ms")
        
        self.validation_results['fk_ik_consistency'] = results
        return results
    
    def analyze_workspace_coverage(self, num_samples: int = 1000, 
                                grid_resolution: int = 20) -> Dict[str, Any]:
        """
        Enhanced workspace coverage analysis using only forward kinematics.
        """
        logger.info(f"Analyzing workspace coverage with {num_samples} samples (FK-only)")
        
        reachable_positions = []
        simple_box_count = 0
        limits_lower, limits_upper = self.fk.get_joint_limits()
        
        # Use only FK - much faster and more reliable
        for _ in range(num_samples):
            q = np.random.uniform(limits_lower, limits_upper, size=self.n_joints)
            T = self.fk.compute_forward_kinematics(q)
            pos = T[:3, 3]
            
            # Check if within simple box bounds
            if (self.workspace_bounds['x_range'][0] <= pos[0] <= self.workspace_bounds['x_range'][1] and
                self.workspace_bounds['y_range'][0] <= pos[1] <= self.workspace_bounds['y_range'][1] and
                self.workspace_bounds['z_range'][0] <= pos[2] <= self.workspace_bounds['z_range'][1]):
                simple_box_count += 1
            
            # Check if within realistic workspace
            if self._realistic_workspace_check(pos):
                reachable_positions.append(pos)
        
        reachable_positions = np.array(reachable_positions)
        
        # Compute workspace statistics
        if len(reachable_positions) > 0:
            workspace_bounds = {
                'x_range': [np.min(reachable_positions[:, 0]), np.max(reachable_positions[:, 0])],
                'y_range': [np.min(reachable_positions[:, 1]), np.max(reachable_positions[:, 1])],
                'z_range': [np.min(reachable_positions[:, 2]), np.max(reachable_positions[:, 2])],
                'volume_estimate': self._estimate_workspace_volume(reachable_positions),
                'centroid': np.mean(reachable_positions, axis=0)
            }
            
            # Additional workspace metrics
            workspace_bounds['xy_coverage'] = self._calculate_xy_coverage(reachable_positions)
            workspace_bounds['z_distribution'] = self._calculate_z_distribution(reachable_positions)
        else:
            workspace_bounds = None
        
        # Enhanced coverage metrics
        coverage_percentage = len(reachable_positions) / num_samples * 100
        realistic_coverage_percentage = len(reachable_positions) / simple_box_count * 100 if simple_box_count > 0 else 0
        
        results = {
            'num_samples': num_samples,
            'reachable_positions': reachable_positions,
            'unreachable_positions': np.array([]),  # Not computed in FK-only mode
            'workspace_bounds': workspace_bounds,
            'coverage_percentage': coverage_percentage,  # Old metric (percentage of all samples)
            'realistic_coverage_percentage': realistic_coverage_percentage,  # New metric (percentage of box-bounded samples)
            'simple_box_count': simple_box_count,
            'grid_coverage': {'resolution': 0, 'total_points': 0, 'reachable_points': 0, 'coverage_ratio': 0.0}
        }
        
        logger.info(f"Workspace Coverage Results (FK-only):")
        logger.info(f"  Raw coverage: {coverage_percentage:.1f}% of random samples")
        logger.info(f"  Realistic coverage: {realistic_coverage_percentage:.1f}% of box-bounded workspace")
        logger.info(f"  Reachable points: {len(reachable_positions)}/{num_samples}")
        
        if workspace_bounds:
            logger.info(f"  X range: [{workspace_bounds['x_range'][0]:.3f}, {workspace_bounds['x_range'][1]:.3f}] m")
            logger.info(f"  Y range: [{workspace_bounds['y_range'][0]:.3f}, {workspace_bounds['y_range'][1]:.3f}] m")
            logger.info(f"  Z range: [{workspace_bounds['z_range'][0]:.3f}, {workspace_bounds['z_range'][1]:.3f}] m")
        
        self.validation_results['workspace_coverage'] = results
        return results
    
    def _calculate_xy_coverage(self, positions: np.ndarray, grid_size: int = 10) -> Dict[str, Any]:
        """Calculate XY plane coverage metrics."""
        if len(positions) == 0:
            return {'coverage_ratio': 0.0}
            
        # Create a grid in XY plane
        x_min, x_max = self.workspace_bounds['x_range']
        y_min, y_max = self.workspace_bounds['y_range']
        
        x_bins = np.linspace(x_min, x_max, grid_size)
        y_bins = np.linspace(y_min, y_max, grid_size)
        
        # Count points in each grid cell
        grid = np.zeros((grid_size-1, grid_size-1))
        for i in range(grid_size-1):
            for j in range(grid_size-1):
                mask = ((positions[:, 0] >= x_bins[i]) & 
                        (positions[:, 0] < x_bins[i+1]) & 
                        (positions[:, 1] >= y_bins[j]) & 
                        (positions[:, 1] < y_bins[j+1]))
                grid[i, j] = np.sum(mask)
        
        # Calculate coverage ratio
        non_zero_cells = np.sum(grid > 0)
        total_cells = (grid_size-1) * (grid_size-1)
        coverage_ratio = non_zero_cells / total_cells
        
        return {
            'grid': grid,
            'coverage_ratio': coverage_ratio,
            'non_zero_cells': non_zero_cells,
            'total_cells': total_cells
        }
    
    def _calculate_z_distribution(self, positions: np.ndarray, bins: int = 10) -> Dict[str, Any]:
        """Analyze height (Z) distribution of reachable positions."""
        if len(positions) == 0:
            return {'distribution': []}
            
        z_min, z_max = self.workspace_bounds['z_range']
        z_hist, z_edges = np.histogram(positions[:, 2], bins=bins, range=(z_min, z_max))
        
        # Normalize
        z_hist = z_hist / np.sum(z_hist)
        
        return {
            'histogram': z_hist.tolist(),
            'bin_edges': z_edges.tolist(),
            'mean_height': np.mean(positions[:, 2]),
            'median_height': np.median(positions[:, 2])
        }

    def benchmark_performance(self, num_fk_tests: int = 1000, 
                            num_ik_tests: int = 100) -> Dict[str, Any]:
        """
        Benchmark forward and inverse kinematics performance.
        
        Args:
            num_fk_tests: Number of FK performance tests
            num_ik_tests: Number of IK performance tests
        """
        logger.info(f"Benchmarking performance (FK: {num_fk_tests}, IK: {num_ik_tests})")
        
        limits_lower, limits_upper = self.fk.get_joint_limits()
        
        # Forward kinematics benchmark
        fk_times = []
        for _ in range(num_fk_tests):
            q = np.random.uniform(limits_lower, limits_upper, size=self.n_joints)
            start_time = time.time()
            T = self.fk.compute_forward_kinematics(q)
            fk_times.append(time.time() - start_time)
        
        # Inverse kinematics benchmark - Use faster parameters for performance testing
        ik_times = []
        ik_success_count = 0
        
        # Fast benchmark parameters (for performance testing only)
        fast_params = {
            'pos_tol': 1e-3,           # 1mm tolerance (acceptable for benchmark)
            'rot_tol': 2e-3,           # Relaxed rotation tolerance
            'max_iters': 100,          # Fewer iterations for speed
            'damping': 5e-4,           # Higher damping for stability
            'step_scale': 0.5,         # Larger steps for faster convergence
            'dq_max': 0.3,             # Allow larger joint steps
            'num_attempts': 15,        # Fewer attempts for speed
            'combined_tolerance': 2e-3, # Relaxed combined tolerance
            'position_relaxation': 0.01, # 10mm relaxation
            'rotation_relaxation': 0.05   # ~3° relaxation
        }
        
        for _ in range(num_ik_tests):
            # Generate target pose
            q_target = np.random.uniform(limits_lower, limits_upper, size=(self.n_joints,))
            T_target = self.fk.compute_forward_kinematics(q_target)
            
            # Only test if within workspace
            if not self._simple_workspace_check(T_target[:3, 3]):
                continue
            
            start_time = time.time()
            q_solution, converged = self.ik.solve(T_target, **fast_params)
            ik_time = time.time() - start_time
            ik_times.append(ik_time)
            
            if converged:
                ik_success_count += 1
        
        results = {
            'fk_performance': {
                'num_tests': num_fk_tests,
                'times': fk_times,
                'mean_time': np.mean(fk_times),
                'std_time': np.std(fk_times),
                'min_time': np.min(fk_times),
                'max_time': np.max(fk_times),
                'frequency_hz': 1.0 / np.mean(fk_times)
            },
            'ik_performance': {
                'num_tests': len(ik_times),
                'success_count': ik_success_count,
                'success_rate': ik_success_count / len(ik_times) if ik_times else 0.0,
                'times': ik_times,
                'mean_time': np.mean(ik_times) if ik_times else 0.0,
                'std_time': np.std(ik_times) if ik_times else 0.0,
                'min_time': np.min(ik_times) if ik_times else 0.0,
                'max_time': np.max(ik_times) if ik_times else 0.0
            }
        }
        
        # Log results
        logger.info(f"Performance Benchmark Results:")
        logger.info(f"  FK mean time: {results['fk_performance']['mean_time']*1000:.3f} ms ({results['fk_performance']['frequency_hz']:.1f} Hz)")
        logger.info(f"  IK mean time: {results['ik_performance']['mean_time']*1000:.1f} ms")
        logger.info(f"  IK success rate: {results['ik_performance']['success_rate']:.1%}")
        
        self.validation_results['performance'] = results
        return results
    
    def validate_against_real_data(self, json_path: str, 
                                 num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate kinematics against real robot data.
        
        Args:
            json_path: Path to JSON file with real robot data
            num_samples: Number of samples to test (None for all)
        """
        logger.info(f"Validating against real robot data: {json_path}")
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load robot data: {e}")
            return {'error': str(e)}
        
        waypoints = data.get('waypoints', [])
        if not waypoints:
            return {'error': 'No waypoints found in data'}
        
        if num_samples and num_samples < len(waypoints):
            indices = np.linspace(0, len(waypoints)-1, num_samples, dtype=int)
            waypoints = [waypoints[i] for i in indices]
        
        position_errors = []
        rotation_errors = []
        ik_test_results = []
        
        for i, wp in enumerate(waypoints):
            try:
                # Extract data
                q_deg = np.array(wp['joint_positions'])
                tcp_recorded_raw = np.array(wp['tcp_position'])
                
                # Convert to standard units
                q_rad, T_recorded = self._convert_from_robot_units(q_deg, tcp_recorded_raw)
                
                # Test forward kinematics
                T_fk = self.fk.compute_forward_kinematics(q_rad)
                
                # Position error
                pos_err = np.linalg.norm(T_fk[:3, 3] - T_recorded[:3, 3])
                position_errors.append(pos_err)
                
                # Rotation error
                R_fk = T_fk[:3, :3]
                R_recorded = T_recorded[:3, :3]
                R_err = R_fk.T @ R_recorded
                cos_angle = np.clip((np.trace(R_err) - 1) / 2.0, -1.0, 1.0)
                rot_err = np.arccos(cos_angle)
                rotation_errors.append(rot_err)
                
                # Test inverse kinematics (every 5th waypoint to save time)
                if i % 5 == 0:
                    q_ik, converged = self.ik.solve(T_recorded, q_init=q_rad)
                    if converged:
                        joint_err = self._compute_joint_error(q_rad, q_ik)
                        ik_test_results.append({
                            'converged': True,
                            'joint_error': joint_err,
                            'original_q': q_rad,
                            'solution_q': q_ik
                        })
                    else:
                        ik_test_results.append({'converged': False})
                
            except Exception as e:
                logger.warning(f"Error processing waypoint {i}: {e}")
        
        # Compute statistics
        results = {
            'num_waypoints': len(waypoints),
            'position_errors': position_errors,
            'rotation_errors': rotation_errors,
            'ik_test_results': ik_test_results
        }
        
        if position_errors:
            results.update({
                'mean_position_error': np.mean(position_errors),
                'max_position_error': np.max(position_errors),
                'std_position_error': np.std(position_errors),
                'mean_rotation_error': np.mean(rotation_errors),
                'max_rotation_error': np.max(rotation_errors),
                'std_rotation_error': np.std(rotation_errors)
            })
        
        if ik_test_results:
            ik_successes = sum(1 for r in ik_test_results if r.get('converged', False))
            results['ik_success_rate'] = ik_successes / len(ik_test_results)
            
            joint_errors = [r['joint_error'] for r in ik_test_results if r.get('converged', False)]
            if joint_errors:
                results['mean_ik_joint_error'] = np.mean(joint_errors)
                results['max_ik_joint_error'] = np.max(joint_errors)
        
        # Log results
        if position_errors:
            logger.info(f"Real Data Validation Results:")
            logger.info(f"  Mean position error: {results['mean_position_error']*1000:.3f} mm")
            logger.info(f"  Mean rotation error: {np.rad2deg(results['mean_rotation_error']):.3f}°")
            if 'ik_success_rate' in results:
                logger.info(f"  IK success rate: {results['ik_success_rate']:.1%}")
        
        self.validation_results['real_data'] = results
        return results
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive validation report."""
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("COMPREHENSIVE ROBOT KINEMATICS VALIDATION REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Robot: RB3-730ES-U (6-DOF)")
        report_lines.append(f"Joints: {self.n_joints}")
        report_lines.append("")
        
        # Overall assessment
        overall_status = self._assess_overall_status()
        report_lines.append("OVERALL ASSESSMENT:")
        report_lines.append(f"  Status: {overall_status['status']}")
        report_lines.append(f"  Score: {overall_status['score']:.1f}/100")
        report_lines.append("")
        
        # Detailed results
        for test_name, result in self.validation_results.items():
            report_lines.extend(self._format_test_result(test_name, result))
            report_lines.append("")
        
        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            report_lines.append("RECOMMENDATIONS:")
            for rec in recommendations:
                report_lines.append(f"  • {rec}")
            report_lines.append("")
        
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        logger.info("Comprehensive validation report generated")
        return report_text
    
    def plot_validation_results(self, save_path: Optional[str] = None) -> Optional[Any]:
        """Create enhanced validation plots with improved workspace visualization."""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available for plotting")
            return None
        
        if 'fk_ik_consistency' not in self.validation_results:
            logger.warning("No FK-IK consistency data available for plotting")
            return None
        
        fk_ik_results = self.validation_results['fk_ik_consistency']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Robot Kinematics Validation Results', fontsize=16)
        
        # Position error histogram
        if fk_ik_results['position_errors']:
            axes[0, 0].hist(np.array(fk_ik_results['position_errors']) * 1000, bins=30, alpha=0.7, color='blue')
            axes[0, 0].set_xlabel('Position Error (mm)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Position Error Distribution')
            axes[0, 0].axvline(fk_ik_results['mean_pos_error'] * 1000, color='red', linestyle='--', label='Mean')
            axes[0, 0].legend()
        
        # Rotation error histogram
        if fk_ik_results['rotation_errors']:
            axes[0, 1].hist(np.rad2deg(fk_ik_results['rotation_errors']), bins=30, alpha=0.7, color='green')
            axes[0, 1].set_xlabel('Rotation Error (degrees)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Rotation Error Distribution')
            axes[0, 1].axvline(np.rad2deg(fk_ik_results['mean_rot_error']), color='red', linestyle='--', label='Mean')
            axes[0, 1].legend()
        
        # Computation time histogram
        if fk_ik_results['computation_times']:
            axes[0, 2].hist(np.array(fk_ik_results['computation_times']) * 1000, bins=30, alpha=0.7, color='orange')
            axes[0, 2].set_xlabel('IK Computation Time (ms)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('IK Performance Distribution')
            axes[0, 2].axvline(fk_ik_results['mean_computation_time'] * 1000, color='red', linestyle='--', label='Mean')
            axes[0, 2].legend()
        
        # Error correlation
        if fk_ik_results['position_errors'] and fk_ik_results['rotation_errors']:
            axes[1, 0].scatter(np.array(fk_ik_results['position_errors']) * 1000, 
                              np.rad2deg(fk_ik_results['rotation_errors']), alpha=0.6, color='purple')
            axes[1, 0].set_xlabel('Position Error (mm)')
            axes[1, 0].set_ylabel('Rotation Error (degrees)')
            axes[1, 0].set_title('Position vs Rotation Error')
        
        # Enhanced workspace visualization with projection views
        if 'workspace_coverage' in self.validation_results:
            ws_results = self.validation_results['workspace_coverage']
            if len(ws_results['reachable_positions']) > 0:
                positions = ws_results['reachable_positions']
                
                # 3D workspace projection (top view)
                axes[1, 1].scatter(positions[:, 0], positions[:, 1], c=positions[:, 2], 
                                  alpha=0.5, s=2, cmap='viridis')
                axes[1, 1].set_xlabel('X (m)')
                axes[1, 1].set_ylabel('Y (m)')
                axes[1, 1].set_title('Workspace Coverage (XY Projection, color=Z)')
                axes[1, 1].axis('equal')
                
                # Draw theoretical workspace boundary circle
                theta = np.linspace(0, 2*np.pi, 100)
                x_circle = self.robot_reach * np.cos(theta)
                y_circle = self.robot_reach * np.sin(theta)
                axes[1, 1].plot(x_circle, y_circle, 'r--', alpha=0.5, label='Max Reach')
                
                # Draw min reach circle
                x_min_circle = self.robot_min_reach * np.cos(theta)
                y_min_circle = self.robot_min_reach * np.sin(theta)
                axes[1, 1].plot(x_min_circle, y_min_circle, 'k--', alpha=0.5, label='Min Reach')
                axes[1, 1].legend(loc='upper right')
        
        # Enhanced summary statistics with workspace metrics
        stats_text = self._format_summary_stats()
        axes[1, 2].text(0.1, 0.5, stats_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center', fontfamily='monospace')
        axes[1, 2].set_title('Validation Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Validation plots saved to: {save_path}")
        
        return fig

    def _compute_joint_error(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Compute joint space error considering periodicity."""
        diff = q2 - q1
        # Handle 2π periodicity for revolute joints
        diff_wrapped = np.mod(diff + np.pi, 2*np.pi) - np.pi
        return np.linalg.norm(diff_wrapped)
    
    def _convert_from_robot_units(self, q_deg: np.ndarray, 
                                tcp_mm_rpy_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert robot units to standard SI units."""
        q_rad = np.deg2rad(q_deg)
        tcp_pos_m = tcp_mm_rpy_deg[:3] / 1000.0
        tcp_rpy_rad = np.deg2rad(tcp_mm_rpy_deg[3:])
        
        R = self.fk.rpy_to_matrix(tcp_rpy_rad)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tcp_pos_m
        
        return q_rad, T
    
    def _estimate_workspace_volume(self, positions: np.ndarray) -> float:
        """Estimate workspace volume using convex hull or bounding box."""
        if len(positions) == 0:
            return 0.0
        
        try:
            # Try to use convex hull if scipy is available
            from scipy.spatial import ConvexHull
            hull = ConvexHull(positions)
            return hull.volume
        except ImportError:
            # Fallback to bounding box volume
            ranges = np.ptp(positions, axis=0)
            return np.prod(ranges)
    
    def _analyze_grid_coverage(self, resolution: int) -> Dict[str, Any]:
        """Analyze workspace coverage using grid-based sampling."""
        # Define workspace bounds for grid
        bounds = {
            'x': [-0.8, 0.8], 'y': [-0.8, 0.8], 'z': [0.1, 1.5]
        }
        
        # Create grid
        x = np.linspace(bounds['x'][0], bounds['x'][1], resolution)
        y = np.linspace(bounds['y'][0], bounds['y'][1], resolution)
        z = np.linspace(bounds['z'][0], bounds['z'][1], resolution)
        
        total_points = 0
        reachable_points = 0
        
        for xi in x:
            for yi in y:
                for zi in z:
                    total_points += 1
                    target_pos = np.array([xi, yi, zi])
                    
                    # Create simple target pose (no specific orientation requirement)
                    T_target = np.eye(4)
                    T_target[:3, 3] = target_pos
                    
                    # Quick IK test
                    q_sol, converged = self.ik.solve(T_target)
                    if converged:
                        reachable_points += 1
        
        return {
            'resolution': resolution,
            'total_points': total_points,
            'reachable_points': reachable_points,
            'coverage_ratio': reachable_points / total_points if total_points > 0 else 0.0
        }
    
    def _assess_overall_status(self) -> Dict[str, Any]:
        """Assess overall validation status."""
        score = 100
        issues = []
        
        # Check screw axes
        if 'screw_axes' in self.validation_results:
            if not self.validation_results['screw_axes']['is_valid']:
                score -= 20
                issues.append("Screw axes verification failed")
        
        # Check FK-IK consistency
        if 'fk_ik_consistency' in self.validation_results:
            success_rate = self.validation_results['fk_ik_consistency']['success_rate']
            if success_rate < 0.90:
                score -= 30
                issues.append(f"Low IK success rate ({success_rate:.1%})")
            elif success_rate < 0.95:
                score -= 10
                issues.append(f"Moderate IK success rate ({success_rate:.1%})")
        
        # Check accuracy
        if 'fk_ik_consistency' in self.validation_results:
            results = self.validation_results['fk_ik_consistency']
            if 'mean_pos_error' in results:
                pos_err_mm = results['mean_pos_error'] * 1000
                if pos_err_mm > 5.0:
                    score -= 20
                    issues.append(f"High position error ({pos_err_mm:.1f} mm)")
                elif pos_err_mm > 1.0:
                    score -= 10
                    issues.append(f"Moderate position error ({pos_err_mm:.1f} mm)")
        
        # Use realistic workspace coverage metric instead of raw coverage
        if 'workspace_coverage' in self.validation_results:
            results = self.validation_results['workspace_coverage']
            if 'realistic_coverage_percentage' in results:
                coverage = results['realistic_coverage_percentage']
                if coverage < 75:
                    score -= 5  # Less severe penalty with realistic metric
                    issues.append(f"Moderate workspace coverage ({coverage:.1f}%)")
            elif 'coverage_percentage' in results:
                coverage = results['coverage_percentage']
                if coverage < 50:
                    score -= 10
                    issues.append(f"Low workspace coverage ({coverage:.1f}%)")
        
        # Determine status
        if score >= 90:
            status = "EXCELLENT - Production Ready"
        elif score >= 70:
            status = "GOOD - Minor Issues"
        elif score >= 50:
            status = "ACCEPTABLE - Needs Improvement"
        else:
            status = "POOR - Major Issues"
        
        return {
            'score': score,
            'status': status,
            'issues': issues
        }
    
    def _format_test_result(self, test_name: str, result: Dict[str, Any]) -> List[str]:
        """Format individual test results for reporting."""
        lines = []
        
        if test_name == 'screw_axes':
            lines.append("SCREW AXES VERIFICATION:")
            lines.append(f"  Status: {'PASS' if result['is_valid'] else 'FAIL'}")
            lines.append(f"  Max difference: {result['max_difference']:.6f}")
            lines.append(f"  RMS error: {result['rms_error']:.6f}")
        
        elif test_name == 'fk_ik_consistency':
            lines.append("FK-IK CONSISTENCY TEST:")
            lines.append(f"  Success rate: {result['success_rate']:.1%} ({result['success_count']}/{result['num_tests']})")
            if 'mean_pos_error' in result:
                lines.append(f"  Mean position error: {result['mean_pos_error']*1000:.3f} mm")
                lines.append(f"  Mean rotation error: {np.rad2deg(result['mean_rot_error']):.3f}°")
                lines.append(f"  Mean computation time: {result['mean_computation_time']*1000:.1f} ms")
        
        elif test_name == 'workspace_coverage':
            lines.append("WORKSPACE COVERAGE ANALYSIS:")
            lines.append(f"  Coverage: {result['coverage_percentage']:.1f}%")
            lines.append(f"  Reachable points: {len(result['reachable_positions'])}/{result['num_samples']}")
            if result['workspace_bounds']:
                bounds = result['workspace_bounds']
                lines.append(f"  Volume estimate: {bounds['volume_estimate']:.3f} m³")
        
        elif test_name == 'real_data':
            lines.append("REAL ROBOT DATA VALIDATION:")
            lines.append(f"  Waypoints tested: {result['num_waypoints']}")
            if 'mean_position_error' in result:
                lines.append(f"  Mean position error: {result['mean_position_error']*1000:.3f} mm")
                lines.append(f"  Mean rotation error: {np.rad2deg(result['mean_rotation_error']):.3f}°")
            if 'ik_success_rate' in result:
                lines.append(f"  IK success rate: {result['ik_success_rate']:.1%}")
        
        return lines
    
    def _format_summary_stats(self) -> str:
        """Format summary statistics for plotting with enhanced workspace metrics."""
        lines = []
        
        if 'fk_ik_consistency' in self.validation_results:
            result = self.validation_results['fk_ik_consistency']
            lines.append(f"Success Rate: {result['success_rate']:.1%}")
            if 'mean_pos_error' in result:
                lines.append(f"Pos Error: {result['mean_pos_error']*1000:.2f} mm")
                lines.append(f"Rot Error: {np.rad2deg(result['mean_rot_error']):.3f}°")
                lines.append(f"IK Time: {result['mean_computation_time']*1000:.1f} ms")
        
        if 'workspace_coverage' in self.validation_results:
            result = self.validation_results['workspace_coverage']
            if 'realistic_coverage_percentage' in result:
                lines.append(f"Realistic Workspace: {result['realistic_coverage_percentage']:.1f}%")
            lines.append(f"Raw Coverage: {result['coverage_percentage']:.1f}%")
            if 'workspace_bounds' in result and result['workspace_bounds']:
                bounds = result['workspace_bounds']
                if 'volume_estimate' in bounds:
                    lines.append(f"Volume: {bounds['volume_estimate']:.3f} m³")
        
        if 'real_data' in self.validation_results:
            result = self.validation_results['real_data']
            if 'mean_position_error' in result:
                lines.append(f"Real Data Pos Err: {result['mean_position_error']*1000:.2f} mm")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check for specific issues and recommend solutions
        if 'fk_ik_consistency' in self.validation_results:
            result = self.validation_results['fk_ik_consistency']
            
            if result['success_rate'] < 0.90:
                recommendations.append("Increase IK random restart attempts")
                recommendations.append("Tune IK convergence tolerances")
            
            if 'mean_pos_error' in result and result['mean_pos_error'] > 0.005:
                recommendations.append("Review robot calibration parameters")
                recommendations.append("Check screw axes computation")
        
        if 'screw_axes' in self.validation_results:
            if not self.validation_results['screw_axes']['is_valid']:
                recommendations.append("Verify robot geometry parameters")
                recommendations.append("Check joint frame transformations")
        
        return recommendations
    
    def _is_pose_reachable(self, pos: np.ndarray) -> bool:
        """More aggressive geometric reachability check."""
        max_reach = 0.650  # Reduced from 0.730 for safety margin
        min_reach = 0.150  # Increased minimum reach
        
        horizontal_dist = np.linalg.norm(pos[:2])
        total_dist = np.linalg.norm(pos)
        
        # More restrictive geometric constraints
        if total_dist > max_reach or total_dist < min_reach:
            return False
        if pos[2] < 0.10 or pos[2] > 1.0:  # Tighter Z bounds
            return False
        
        # Avoid extreme orientations near workspace boundaries
        if horizontal_dist > 0.5 and pos[2] < 0.3:  # Avoid low + far poses
            return False
            
        return True
    



# Convenience function for comprehensive validation
def run_comprehensive_validation(robot_controller, 
                               num_fk_ik_tests: int = 200,
                               num_workspace_samples: int = 1000,
                               real_data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Run comprehensive validation with the new modular structure.
    
    Args:
        robot_controller: RobotController instance
        num_fk_ik_tests: Number of FK-IK consistency tests
        num_workspace_samples: Number of workspace coverage samples
        real_data_path: Path to real robot data (optional)
    """
    from .forward_kinematic import ForwardKinematics
    from .inverse_kinematic import FastIK
    
    # Create kinematics modules
    fk = ForwardKinematics()
    ik = FastIK(fk)
    
    # Create validator
    validator = KinematicsValidator(fk, ik)
    
    logger.info("Starting comprehensive kinematics validation")
    
    # Run all validation tests
    results = {}
    
    # 1. Screw axes verification
    results['screw_axes'] = validator.verify_screw_axes_theory()
    
    # 2. FK-IK consistency
    results['fk_ik_consistency'] = validator.test_fk_ik_consistency(num_fk_ik_tests)
    
    # 3. Workspace coverage
    results['workspace_coverage'] = validator.analyze_workspace_coverage(num_workspace_samples)
    
    # 4. Performance benchmark
    results['performance'] = validator.benchmark_performance()
    
    # 5. Real data validation (if available)
    if real_data_path and os.path.exists(real_data_path):
        results['real_data'] = validator.validate_against_real_data(real_data_path)
    
    # Generate comprehensive report
    report = validator.generate_comprehensive_report()
    print(report)
    
    # Create plots
    validator.plot_validation_results('comprehensive_validation.png')
    
    logger.info("Comprehensive validation completed")
    return results