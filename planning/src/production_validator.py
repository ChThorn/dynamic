#!/usr/bin/env python3
"""
Production Validation System for Planning Module

Provides comprehensive validation, diagnostics, and production-ready
features for the motion planning system.

Author: Robot Control Team
"""

import numpy as np
import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import yaml

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation strictness levels."""
    DEVELOPMENT = "development"    # Relaxed validation for testing
    PRODUCTION = "production"      # Strict validation for deployment
    SAFETY_CRITICAL = "safety_critical"  # Maximum validation for critical applications

class SystemStatus(Enum):
    """System health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of system validation."""
    status: SystemStatus
    level: ValidationLevel
    checks_passed: int
    checks_total: int
    errors: List[str]
    warnings: List[str]
    performance_metrics: Dict[str, float]
    recommendations: List[str]
    timestamp: float

@dataclass
class ProductionReport:
    """Comprehensive production readiness report."""
    overall_status: SystemStatus
    module_statuses: Dict[str, SystemStatus]
    validation_results: Dict[str, ValidationResult]
    configuration_issues: List[str]
    performance_summary: Dict[str, Any]
    deployment_recommendations: List[str]
    generated_at: float

class ProductionValidator:
    """
    Production validation system for motion planning modules.
    
    Provides comprehensive testing, diagnostics, and health monitoring
    for production deployment of the robot motion planning system.
    """
    
    def __init__(self, config_path: str, validation_level: ValidationLevel = ValidationLevel.PRODUCTION):
        """Initialize production validator."""
        self.config_path = config_path
        self.validation_level = validation_level
        self.load_configuration()
        
        # Validation thresholds based on level
        self.thresholds = self._get_validation_thresholds()
        
        logger.info(f"Production validator initialized: {validation_level.value} level")
    
    def load_configuration(self):
        """Load system configuration."""
        try:
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            logger.info("Production validator configuration loaded")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = {}
    
    def _get_validation_thresholds(self) -> Dict[str, Any]:
        """Get validation thresholds based on validation level."""
        base_thresholds = {
            'max_planning_time': 5.0,      # seconds
            'min_success_rate': 0.85,      # 85%
            'max_collision_false_positive': 0.1,  # 10%
            'min_ik_convergence': 0.8,     # 80%
            'max_trajectory_error': 0.005,  # 5mm
            'min_workspace_coverage': 0.7   # 70%
        }
        
        if self.validation_level == ValidationLevel.DEVELOPMENT:
            # Relaxed thresholds for development
            return {k: v * 0.7 for k, v in base_thresholds.items()}
        elif self.validation_level == ValidationLevel.SAFETY_CRITICAL:
            # Stricter thresholds for safety-critical applications
            return {k: v * 1.3 for k, v in base_thresholds.items()}
        else:
            # Production thresholds
            return base_thresholds
    
    def validate_collision_system(self) -> ValidationResult:
        """Validate collision detection system."""
        start_time = time.time()
        errors = []
        warnings = []
        checks_passed = 0
        checks_total = 6
        
        try:
            from collision_checker import EnhancedCollisionChecker
            checker = EnhancedCollisionChecker(self.config_path)
            
            # Test 1: Configuration loading
            if hasattr(checker, 'config') and checker.config:
                checks_passed += 1
            else:
                errors.append("Collision checker configuration not loaded properly")
            
            # Test 2: Critical joint pairs defined
            if hasattr(checker, 'critical_joint_pairs') and len(checker.critical_joint_pairs) > 0:
                checks_passed += 1
            else:
                errors.append("Critical joint pairs not defined")
            
            # Test 3: Minimum distance thresholds reasonable
            if hasattr(checker, 'min_joint_distances'):
                reasonable_thresholds = all(
                    0.010 <= dist <= 0.100  # 10mm to 100mm range
                    for dist in checker.min_joint_distances.values()
                )
                if reasonable_thresholds:
                    checks_passed += 1
                else:
                    warnings.append("Some collision thresholds may be unrealistic")
            else:
                errors.append("Minimum distance thresholds not configured")
            
            # Test 4: Home position validation
            try:
                home_pos = np.zeros(6)
                home_tcp = np.array([0, 0, 0.8])
                result = checker.check_configuration_collision(home_pos, home_tcp)
                if not result.is_collision:
                    checks_passed += 1
                else:
                    errors.append(f"Home position incorrectly flagged as collision: {result.details}")
            except Exception as e:
                errors.append(f"Home position validation failed: {e}")
            
            # Test 5: Adaptive threshold functionality
            try:
                test_config = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
                for pair in checker.critical_joint_pairs[:2]:  # Test first 2 pairs
                    threshold = checker._get_adaptive_threshold(pair[0], pair[1], test_config)
                    if threshold > 5.0:  # Minimum 5mm threshold
                        checks_passed += 1
                        break
                else:
                    warnings.append("Adaptive thresholds may be too small")
            except Exception as e:
                errors.append(f"Adaptive threshold test failed: {e}")
            
            # Test 6: Performance test
            validation_time = time.time() - start_time
            if validation_time < 1.0:  # Should complete within 1 second
                checks_passed += 1
            else:
                warnings.append("Collision validation performance slower than expected")
            
        except Exception as e:
            errors.append(f"Collision system initialization failed: {e}")
        
        # Determine status
        if len(errors) == 0 and checks_passed >= checks_total * 0.9:
            status = SystemStatus.HEALTHY
        elif len(errors) == 0 and checks_passed >= checks_total * 0.7:
            status = SystemStatus.WARNING
        else:
            status = SystemStatus.ERROR
        
        return ValidationResult(
            status=status,
            level=self.validation_level,
            checks_passed=checks_passed,
            checks_total=checks_total,
            errors=errors,
            warnings=warnings,
            performance_metrics={'validation_time': validation_time},
            recommendations=self._get_collision_recommendations(errors, warnings),
            timestamp=time.time()
        )
    
    def validate_motion_planner(self) -> ValidationResult:
        """Validate motion planning system."""
        start_time = time.time()
        errors = []
        warnings = []
        checks_passed = 0
        checks_total = 8
        
        try:
            # Import required modules
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'kinematics', 'src'))
            
            from forward_kinematic import ForwardKinematics
            from inverse_kinematic import InverseKinematics
            from motion_planner import MotionPlanner, PlanningStrategy, PlanningStatus
            
            # Initialize system
            fk = ForwardKinematics()
            ik = InverseKinematics(fk)
            planner = MotionPlanner(fk, ik)
            
            # Test 1: System initialization
            if planner and hasattr(planner, 'config'):
                checks_passed += 1
            else:
                errors.append("Motion planner initialization failed")
            
            # Test 2: Thread safety
            if hasattr(planner, '_planning_lock') and hasattr(planner, '_stats_lock'):
                checks_passed += 1
            else:
                warnings.append("Thread safety locks not properly initialized")
            
            # Test 3: Basic IK test
            try:
                test_pose = np.eye(4)
                test_pose[:3, 3] = [0.4, 0.0, 0.5]  # Reachable position
                q_solution, ik_success = ik.solve(test_pose)
                if ik_success:
                    checks_passed += 1
                else:
                    warnings.append("Basic IK test failed - may indicate configuration issues")
            except Exception as e:
                warnings.append(f"IK test error: {e}")
            
            # Test 4: Joint space planning
            try:
                start_config = np.zeros(6)
                goal_config = np.array([0.0, -0.5, 0.5, 0.0, 0.0, 0.0])
                result = planner.plan_motion(start_config, goal_config, strategy=PlanningStrategy.JOINT_SPACE)
                if result.status == PlanningStatus.SUCCESS:
                    checks_passed += 1
                else:
                    warnings.append(f"Joint space planning test failed: {result.error_message}")
            except Exception as e:
                warnings.append(f"Joint space planning error: {e}")
            
            # Test 5: Configuration updates
            try:
                test_config = {'test_param': 123}
                planner.update_config(test_config)
                if 'test_param' in planner.config and planner.config['test_param'] == 123:
                    checks_passed += 1
                else:
                    warnings.append("Configuration update test failed")
            except Exception as e:
                warnings.append(f"Configuration update error: {e}")
            
            # Test 6: Statistics tracking
            try:
                stats = planner.get_statistics()
                if isinstance(stats, dict) and 'total_plans' in stats:
                    checks_passed += 1
                else:
                    warnings.append("Statistics tracking not working properly")
            except Exception as e:
                warnings.append(f"Statistics tracking error: {e}")
            
            # Test 7: Fallback strategies
            if planner.config.get('enable_fallbacks', False):
                checks_passed += 1
            else:
                warnings.append("Fallback strategies not enabled")
            
            # Test 8: Performance test
            validation_time = time.time() - start_time
            if validation_time < self.thresholds['max_planning_time']:
                checks_passed += 1
            else:
                warnings.append("Motion planner validation performance slower than expected")
            
        except Exception as e:
            errors.append(f"Motion planner system validation failed: {e}")
        
        # Determine status
        if len(errors) == 0 and checks_passed >= checks_total * 0.9:
            status = SystemStatus.HEALTHY
        elif len(errors) == 0 and checks_passed >= checks_total * 0.7:
            status = SystemStatus.WARNING
        else:
            status = SystemStatus.ERROR
        
        return ValidationResult(
            status=status,
            level=self.validation_level,
            checks_passed=checks_passed,
            checks_total=checks_total,
            errors=errors,
            warnings=warnings,
            performance_metrics={'validation_time': validation_time},
            recommendations=self._get_motion_planner_recommendations(errors, warnings),
            timestamp=time.time()
        )
    
    def validate_trajectory_planner(self) -> ValidationResult:
        """Validate trajectory planning system."""
        start_time = time.time()
        errors = []
        warnings = []
        checks_passed = 0
        checks_total = 6
        
        try:
            from trajectory_planner import TrajectoryPlanner
            
            planner = TrajectoryPlanner()
            
            # Test 1: Initialization
            if planner and hasattr(planner, 'config'):
                checks_passed += 1
            else:
                errors.append("Trajectory planner initialization failed")
            
            # Test 2: Configuration validation
            required_configs = ['max_joint_velocity', 'max_joint_acceleration', 'time_resolution']
            if all(key in planner.config for key in required_configs):
                checks_passed += 1
            else:
                warnings.append("Missing required trajectory configuration parameters")
            
            # Test 3: Basic trajectory generation
            try:
                waypoints = [
                    np.zeros(6),
                    np.array([0.0, -0.3, 0.3, 0.0, 0.0, 0.0]),
                    np.zeros(6)
                ]
                result = planner.plan_trajectory(waypoints)
                if result.success:
                    checks_passed += 1
                else:
                    warnings.append(f"Basic trajectory generation failed: {result.error_message}")
            except Exception as e:
                warnings.append(f"Trajectory generation error: {e}")
            
            # Test 4: Time scaling
            try:
                waypoints = [np.zeros(6), np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])]
                result1 = planner.plan_trajectory(waypoints, time_scaling=1.0)
                result2 = planner.plan_trajectory(waypoints, time_scaling=2.0)
                
                if (result1.success and result2.success and 
                    result2.trajectory.total_time > result1.trajectory.total_time * 1.5):
                    checks_passed += 1
                else:
                    warnings.append("Time scaling functionality not working properly")
            except Exception as e:
                warnings.append(f"Time scaling test error: {e}")
            
            # Test 5: Constraint validation
            vel_limit = planner.config['max_joint_velocity']
            acc_limit = planner.config['max_joint_acceleration']
            
            if vel_limit > 0 and acc_limit > 0:
                checks_passed += 1
            else:
                errors.append("Invalid velocity or acceleration limits")
            
            # Test 6: Performance test
            validation_time = time.time() - start_time
            if validation_time < 2.0:  # Should complete within 2 seconds
                checks_passed += 1
            else:
                warnings.append("Trajectory planner validation performance slower than expected")
            
        except Exception as e:
            errors.append(f"Trajectory planner validation failed: {e}")
        
        # Determine status
        if len(errors) == 0 and checks_passed >= checks_total * 0.9:
            status = SystemStatus.HEALTHY
        elif len(errors) == 0 and checks_passed >= checks_total * 0.7:
            status = SystemStatus.WARNING
        else:
            status = SystemStatus.ERROR
        
        return ValidationResult(
            status=status,
            level=self.validation_level,
            checks_passed=checks_passed,
            checks_total=checks_total,
            errors=errors,
            warnings=warnings,
            performance_metrics={'validation_time': validation_time},
            recommendations=self._get_trajectory_recommendations(errors, warnings),
            timestamp=time.time()
        )
    
    def generate_production_report(self) -> ProductionReport:
        """Generate comprehensive production readiness report."""
        logger.info("Generating production readiness report...")
        
        # Validate all systems
        collision_result = self.validate_collision_system()
        motion_result = self.validate_motion_planner()
        trajectory_result = self.validate_trajectory_planner()
        
        # Compile results
        validation_results = {
            'collision_system': collision_result,
            'motion_planner': motion_result,
            'trajectory_planner': trajectory_result
        }
        
        module_statuses = {
            'collision_system': collision_result.status,
            'motion_planner': motion_result.status,
            'trajectory_planner': trajectory_result.status
        }
        
        # Determine overall status
        statuses = list(module_statuses.values())
        if all(s == SystemStatus.HEALTHY for s in statuses):
            overall_status = SystemStatus.HEALTHY
        elif any(s == SystemStatus.CRITICAL for s in statuses):
            overall_status = SystemStatus.CRITICAL
        elif any(s == SystemStatus.ERROR for s in statuses):
            overall_status = SystemStatus.ERROR
        else:
            overall_status = SystemStatus.WARNING
        
        # Compile configuration issues
        config_issues = []
        for result in validation_results.values():
            config_issues.extend(result.errors)
        
        # Performance summary
        performance_summary = {
            'total_checks': sum(r.checks_total for r in validation_results.values()),
            'passed_checks': sum(r.checks_passed for r in validation_results.values()),
            'total_validation_time': sum(
                r.performance_metrics.get('validation_time', 0) 
                for r in validation_results.values()
            ),
            'success_rate': sum(r.checks_passed for r in validation_results.values()) / 
                          sum(r.checks_total for r in validation_results.values())
        }
        
        # Deployment recommendations
        deployment_recommendations = []
        for result in validation_results.values():
            deployment_recommendations.extend(result.recommendations)
        
        # Add overall recommendations based on status
        if overall_status == SystemStatus.HEALTHY:
            deployment_recommendations.append("✅ System ready for production deployment")
        elif overall_status == SystemStatus.WARNING:
            deployment_recommendations.append("⚠️ System can be deployed with monitoring - address warnings")
        elif overall_status == SystemStatus.ERROR:
            deployment_recommendations.append("❌ System NOT ready for production - fix errors first")
        else:
            deployment_recommendations.append("🚨 CRITICAL issues found - immediate attention required")
        
        return ProductionReport(
            overall_status=overall_status,
            module_statuses=module_statuses,
            validation_results=validation_results,
            configuration_issues=config_issues,
            performance_summary=performance_summary,
            deployment_recommendations=deployment_recommendations,
            generated_at=time.time()
        )
    
    def _get_collision_recommendations(self, errors: List[str], warnings: List[str]) -> List[str]:
        """Get recommendations for collision system issues."""
        recommendations = []
        
        if any("Home position" in error for error in errors):
            recommendations.append("Recalibrate collision detection thresholds in constraints.yaml")
            recommendations.append("Verify minimum distance values are realistic for RB3-730ES-U")
        
        if any("threshold" in warning.lower() for warning in warnings):
            recommendations.append("Review adaptive threshold calculation algorithm")
            recommendations.append("Ensure minimum threshold enforcement is working")
        
        if any("performance" in warning.lower() for warning in warnings):
            recommendations.append("Optimize collision checking algorithms for better performance")
        
        return recommendations
    
    def _get_motion_planner_recommendations(self, errors: List[str], warnings: List[str]) -> List[str]:
        """Get recommendations for motion planner issues."""
        recommendations = []
        
        if any("IK" in warning for warning in warnings):
            recommendations.append("Enable C-space analysis for better IK convergence")
            recommendations.append("Review IK solver configuration and initial guess strategies")
        
        if any("planning" in warning.lower() for warning in warnings):
            recommendations.append("Check collision detection settings affecting motion planning")
            recommendations.append("Consider enabling fallback planning strategies")
        
        if any("thread" in warning.lower() for warning in warnings):
            recommendations.append("Verify thread safety implementation for production use")
        
        return recommendations
    
    def _get_trajectory_recommendations(self, errors: List[str], warnings: List[str]) -> List[str]:
        """Get recommendations for trajectory planner issues."""
        recommendations = []
        
        if any("limit" in error.lower() for error in errors):
            recommendations.append("Update velocity and acceleration limits in trajectory configuration")
        
        if any("scaling" in warning.lower() for warning in warnings):
            recommendations.append("Review time scaling algorithm implementation")
        
        if any("performance" in warning.lower() for warning in warnings):
            recommendations.append("Optimize trajectory generation algorithms")
        
        return recommendations
    
    def save_report(self, report: ProductionReport, filepath: str) -> bool:
        """Save production report to file."""
        try:
            # Convert dataclasses to dictionaries for JSON serialization
            report_dict = asdict(report)
            
            # Convert enums to strings
            def convert_enums(obj):
                if isinstance(obj, dict):
                    return {k: convert_enums(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_enums(item) for item in obj]
                elif isinstance(obj, Enum):
                    return obj.value
                else:
                    return obj
            
            report_dict = convert_enums(report_dict)
            
            with open(filepath, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            logger.info(f"Production report saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save production report: {e}")
            return False


def main():
    """Main validation function for command-line use."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production validation for motion planning system")
    parser.add_argument("--config", default="../config/constraints.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--level", choices=['development', 'production', 'safety_critical'],
                       default='production', help="Validation level")
    parser.add_argument("--output", help="Output file for report")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create validator
    validation_level = ValidationLevel(args.level)
    validator = ProductionValidator(args.config, validation_level)
    
    # Generate report
    report = validator.generate_production_report()
    
    # Print summary
    print(f"\n{'='*60}")
    print("PRODUCTION READINESS REPORT")
    print(f"{'='*60}")
    print(f"Overall Status: {report.overall_status.value.upper()}")
    print(f"Validation Level: {validation_level.value}")
    print(f"Total Checks: {report.performance_summary['passed_checks']}/{report.performance_summary['total_checks']}")
    print(f"Success Rate: {report.performance_summary['success_rate']*100:.1f}%")
    
    print(f"\nModule Status:")
    for module, status in report.module_statuses.items():
        print(f"  {module}: {status.value.upper()}")
    
    if report.configuration_issues:
        print(f"\nConfiguration Issues:")
        for issue in report.configuration_issues:
            print(f"  - {issue}")
    
    print(f"\nDeployment Recommendations:")
    for rec in report.deployment_recommendations:
        print(f"  - {rec}")
    
    # Save report if requested
    if args.output:
        if validator.save_report(report, args.output):
            print(f"\nDetailed report saved to: {args.output}")
    
    # Exit with appropriate code
    if report.overall_status in [SystemStatus.HEALTHY, SystemStatus.WARNING]:
        exit(0)
    else:
        exit(1)


if __name__ == "__main__":
    main()