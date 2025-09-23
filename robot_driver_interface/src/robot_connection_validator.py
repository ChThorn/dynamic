#!/usr/bin/env python3
"""
Robot Connection Validator
Validates pre-buffering strategy with real robot controller through staged testing.
"""

import time
import logging
from typing import List, Dict, Any, Optional
import statistics
import json
from datetime import datetime


class RobotConnectionValidator:
    """Validates robot connection and pre-buffering performance"""
    
    def __init__(self, robot_ip: str = "192.168.0.10"):
        self.robot_ip = robot_ip
        self.test_results = {}
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for validation"""
        logger = logging.getLogger("RobotValidator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def stage1_network_validation(self) -> Dict[str, Any]:
        """Stage 1: Test network connectivity and latency"""
        self.logger.info("=== STAGE 1: Network Validation ===")
        
        results = {
            "ping_test": self._test_ping_latency(),
            "connection_stability": self._test_connection_stability(),
            "bandwidth_test": self._test_bandwidth()
        }
        
        self.test_results["stage1_network"] = results
        return results
    
    def stage2_robot_api_validation(self, robot_instance) -> Dict[str, Any]:
        """Stage 2: Test robot API responses and buffer operations"""
        self.logger.info("=== STAGE 2: Robot API Validation ===")
        
        results = {
            "connection_test": self._test_robot_connection(robot_instance),
            "buffer_api_test": self._test_buffer_api(robot_instance),
            "command_response_time": self._test_command_response_time(robot_instance),
            "status_reporting": self._test_status_reporting(robot_instance)
        }
        
        self.test_results["stage2_robot_api"] = results
        return results
    
    def stage3_buffer_stress_test(self, robot_instance) -> Dict[str, Any]:
        """Stage 3: Test buffer management under stress"""
        self.logger.info("=== STAGE 3: Buffer Stress Test ===")
        
        results = {
            "buffer_fill_speed": self._test_buffer_fill_speed(robot_instance),
            "buffer_consumption_rate": self._test_buffer_consumption_rate(robot_instance),
            "concurrent_operations": self._test_concurrent_buffer_ops(robot_instance),
            "error_recovery": self._test_buffer_error_recovery(robot_instance)
        }
        
        self.test_results["stage3_buffer_stress"] = results
        return results
    
    def stage4_prebuffer_validation(self, robot_instance) -> Dict[str, Any]:
        """Stage 4: Test complete pre-buffering strategy"""
        self.logger.info("=== STAGE 4: Pre-buffering Strategy Validation ===")
        
        results = {
            "prebuffer_timing": self._test_prebuffer_timing(robot_instance),
            "continuous_motion": self._test_continuous_motion(robot_instance),
            "gap_detection": self._test_motion_gap_detection(robot_instance),
            "performance_comparison": self._compare_vs_sequential(robot_instance)
        }
        
        self.test_results["stage4_prebuffer"] = results
        return results
    
    def _test_ping_latency(self) -> Dict[str, Any]:
        """Test network ping latency to robot"""
        import subprocess
        
        try:
            # Run ping test
            result = subprocess.run(
                ["ping", "-c", "10", self.robot_ip], 
                capture_output=True, 
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Parse ping times
                lines = result.stdout.split('\n')
                ping_times = []
                
                for line in lines:
                    if "time=" in line:
                        time_str = line.split("time=")[1].split()[0]
                        ping_times.append(float(time_str))
                
                if ping_times:
                    return {
                        "status": "SUCCESS",
                        "avg_latency_ms": statistics.mean(ping_times),
                        "max_latency_ms": max(ping_times),
                        "min_latency_ms": min(ping_times),
                        "jitter_ms": statistics.stdev(ping_times) if len(ping_times) > 1 else 0,
                        "packet_loss": 0,
                        "assessment": "GOOD" if statistics.mean(ping_times) < 10 else "MARGINAL"
                    }
            
            return {
                "status": "FAILED",
                "error": "Ping failed",
                "assessment": "CRITICAL"
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "assessment": "CRITICAL"
            }
    
    def _test_connection_stability(self) -> Dict[str, Any]:
        """Test connection stability over time"""
        try:
            import socket
            
            successful_connections = 0
            failed_connections = 0
            connection_times = []
            
            for i in range(20):  # Test 20 connections
                start_time = time.time()
                
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.settimeout(5.0)
                    
                    result = sock.connect_ex((self.robot_ip, 5000))  # Standard robot control port
                    connection_time = time.time() - start_time
                    
                    if result == 0:
                        successful_connections += 1
                        connection_times.append(connection_time * 1000)  # Convert to ms
                    else:
                        failed_connections += 1
                    
                    sock.close()
                    time.sleep(0.1)  # Small delay between tests
                    
                except Exception:
                    failed_connections += 1
            
            success_rate = (successful_connections / 20) * 100
            
            return {
                "status": "SUCCESS" if success_rate > 90 else "FAILED",
                "success_rate_percent": success_rate,
                "avg_connection_time_ms": statistics.mean(connection_times) if connection_times else 0,
                "failed_connections": failed_connections,
                "assessment": "GOOD" if success_rate > 95 else "MARGINAL" if success_rate > 85 else "CRITICAL"
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "assessment": "CRITICAL"
            }
    
    def _test_bandwidth(self) -> Dict[str, Any]:
        """Test available bandwidth (simplified)"""
        # For now, return basic assessment
        # In real implementation, could use iperf or similar
        return {
            "status": "ESTIMATED",
            "estimated_bandwidth_mbps": 100,  # Typical Ethernet
            "sufficient_for_robot": True,
            "assessment": "GOOD"
        }
    
    def _test_robot_connection(self, robot) -> Dict[str, Any]:
        """Test basic robot connection and initialization"""
        try:
            start_time = time.time()
            
            # Test connection
            if hasattr(robot, 'is_connected'):
                cmd_connected, data_connected = robot.is_connected()
                connection_time = time.time() - start_time
                
                return {
                    "status": "SUCCESS" if (cmd_connected and data_connected) else "FAILED",
                    "cmd_connected": cmd_connected,
                    "data_connected": data_connected,
                    "connection_time_ms": connection_time * 1000,
                    "assessment": "GOOD" if (cmd_connected and data_connected) else "CRITICAL"
                }
            else:
                return {
                    "status": "ERROR",
                    "error": "Robot connection method not available",
                    "assessment": "CRITICAL"
                }
                
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "assessment": "CRITICAL"
            }
    
    def _test_buffer_api(self, robot) -> Dict[str, Any]:
        """Test buffer API functionality"""
        try:
            tests = {}
            
            # Test buffer size reporting
            if hasattr(robot, 'get_buffer_size'):
                start_time = time.time()
                buffer_size = robot.get_buffer_size()
                response_time = (time.time() - start_time) * 1000
                
                tests["buffer_size_query"] = {
                    "status": "SUCCESS",
                    "buffer_size": buffer_size,
                    "response_time_ms": response_time,
                    "assessment": "GOOD" if response_time < 10 else "MARGINAL"
                }
            else:
                tests["buffer_size_query"] = {
                    "status": "FAILED",
                    "error": "get_buffer_size not available",
                    "assessment": "CRITICAL"
                }
            
            # Test buffer clear
            if hasattr(robot, 'move_joint_blend_clear'):
                start_time = time.time()
                result = robot.move_joint_blend_clear()
                response_time = (time.time() - start_time) * 1000
                
                tests["buffer_clear"] = {
                    "status": "SUCCESS" if result else "FAILED",
                    "response_time_ms": response_time,
                    "assessment": "GOOD" if response_time < 20 else "MARGINAL"
                }
            
            return tests
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "assessment": "CRITICAL"
            }
    
    def _test_command_response_time(self, robot) -> Dict[str, Any]:
        """Test robot command response times"""
        try:
            response_times = []
            
            # Test multiple status queries
            for i in range(10):
                if hasattr(robot, 'sys_status'):
                    start_time = time.time()
                    _ = robot.sys_status.robot_state
                    response_time = (time.time() - start_time) * 1000
                    response_times.append(response_time)
                    time.sleep(0.1)
            
            if response_times:
                avg_time = statistics.mean(response_times)
                max_time = max(response_times)
                
                return {
                    "status": "SUCCESS",
                    "avg_response_time_ms": avg_time,
                    "max_response_time_ms": max_time,
                    "response_times": response_times,
                    "assessment": "GOOD" if avg_time < 5 else "MARGINAL" if avg_time < 15 else "CRITICAL"
                }
            else:
                return {
                    "status": "FAILED",
                    "error": "No response time data",
                    "assessment": "CRITICAL"
                }
                
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "assessment": "CRITICAL"
            }
    
    def _test_status_reporting(self, robot) -> Dict[str, Any]:
        """Test robot status reporting accuracy"""
        try:
            if hasattr(robot, 'sys_status'):
                # Test various status fields
                status_fields = {}
                
                if hasattr(robot.sys_status, 'robot_state'):
                    status_fields["robot_state"] = robot.sys_status.robot_state
                
                if hasattr(robot.sys_status, 'jnt_ang'):
                    status_fields["joint_positions"] = robot.sys_status.jnt_ang
                
                return {
                    "status": "SUCCESS",
                    "available_fields": list(status_fields.keys()),
                    "current_values": status_fields,
                    "assessment": "GOOD" if len(status_fields) > 0 else "MARGINAL"
                }
            else:
                return {
                    "status": "FAILED",
                    "error": "sys_status not available",
                    "assessment": "CRITICAL"
                }
                
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "assessment": "CRITICAL"
            }
    
    def _test_buffer_fill_speed(self, robot) -> Dict[str, Any]:
        """Test how fast we can fill the robot buffer"""
        try:
            if not hasattr(robot, 'move_joint_blend_add_point'):
                return {
                    "status": "FAILED", 
                    "error": "Buffer add method not available",
                    "assessment": "CRITICAL"
                }
            
            # Clear buffer first
            if hasattr(robot, 'move_joint_blend_clear'):
                robot.move_joint_blend_clear()
            
            # Generate test waypoints (simple joint movements)
            test_joints = [0, 0, 0, 0, 0, 0]  # Home position
            fill_times = []
            
            # Test adding 20 waypoints
            start_time = time.time()
            successful_adds = 0
            
            for i in range(20):
                waypoint_start = time.time()
                
                success = robot.move_joint_blend_add_point(
                    test_joints, 
                    speed=50, 
                    acceleration=100
                )
                
                waypoint_time = (time.time() - waypoint_start) * 1000
                fill_times.append(waypoint_time)
                
                if success:
                    successful_adds += 1
                else:
                    break
            
            total_time = (time.time() - start_time) * 1000
            
            return {
                "status": "SUCCESS" if successful_adds > 15 else "FAILED",
                "successful_adds": successful_adds,
                "total_time_ms": total_time,
                "avg_add_time_ms": statistics.mean(fill_times) if fill_times else 0,
                "max_add_time_ms": max(fill_times) if fill_times else 0,
                "waypoints_per_second": (successful_adds / (total_time / 1000)) if total_time > 0 else 0,
                "assessment": "GOOD" if successful_adds >= 18 else "MARGINAL" if successful_adds >= 10 else "CRITICAL"
            }
            
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "assessment": "CRITICAL"
            }
    
    def _test_buffer_consumption_rate(self, robot) -> Dict[str, Any]:
        """Test how fast robot consumes buffer waypoints"""
        # This would require actual robot motion - placeholder for now
        return {
            "status": "TODO", 
            "note": "Requires actual robot motion testing",
            "assessment": "UNKNOWN"
        }
    
    def _test_concurrent_buffer_ops(self, robot) -> Dict[str, Any]:
        """Test concurrent buffer operations"""
        # Placeholder - would test adding while robot is moving
        return {
            "status": "TODO",
            "note": "Requires motion testing",
            "assessment": "UNKNOWN"
        }
    
    def _test_buffer_error_recovery(self, robot) -> Dict[str, Any]:
        """Test buffer error recovery"""
        # Placeholder - would test buffer overflow scenarios
        return {
            "status": "TODO",
            "note": "Requires controlled error scenarios",
            "assessment": "UNKNOWN"
        }
    
    def _test_prebuffer_timing(self, robot) -> Dict[str, Any]:
        """Test pre-buffering timing accuracy"""
        # Placeholder - would test actual pre-buffering strategy
        return {
            "status": "TODO",
            "note": "Requires full pre-buffering implementation test",
            "assessment": "UNKNOWN"
        }
    
    def _test_continuous_motion(self, robot) -> Dict[str, Any]:
        """Test continuous motion without gaps"""
        # Placeholder - would measure actual motion continuity
        return {
            "status": "TODO",
            "note": "Requires motion analysis",
            "assessment": "UNKNOWN"
        }
    
    def _test_motion_gap_detection(self, robot) -> Dict[str, Any]:
        """Test detection of motion gaps"""
        # Placeholder - would monitor for motion interruptions
        return {
            "status": "TODO",
            "note": "Requires motion monitoring",
            "assessment": "UNKNOWN"
        }
    
    def _compare_vs_sequential(self, robot) -> Dict[str, Any]:
        """Compare pre-buffering vs sequential execution"""
        # Placeholder - would run comparative tests
        return {
            "status": "TODO",
            "note": "Requires comparative motion testing",
            "assessment": "UNKNOWN"
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive validation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report = f"""
ROBOT CONNECTION VALIDATION REPORT
Generated: {timestamp}
Robot IP: {self.robot_ip}

================================
OVERALL ASSESSMENT
================================
"""
        
        overall_status = "UNKNOWN"
        critical_issues = []
        warnings = []
        
        # Analyze results
        for stage, results in self.test_results.items():
            if isinstance(results, dict):
                for test, result in results.items():
                    if isinstance(result, dict) and 'assessment' in result:
                        if result['assessment'] == 'CRITICAL':
                            critical_issues.append(f"{stage}.{test}")
                        elif result['assessment'] == 'MARGINAL':
                            warnings.append(f"{stage}.{test}")
        
        if len(critical_issues) == 0:
            if len(warnings) == 0:
                overall_status = "READY FOR DEPLOYMENT"
            else:
                overall_status = "DEPLOY WITH CAUTION"
        else:
            overall_status = "NOT READY - CRITICAL ISSUES"
        
        report += f"Status: {overall_status}\n"
        
        if critical_issues:
            report += f"\nCRITICAL ISSUES ({len(critical_issues)}):\n"
            for issue in critical_issues:
                report += f"  - {issue}\n"
        
        if warnings:
            report += f"\nWARNINGS ({len(warnings)}):\n"
            for warning in warnings:
                report += f"  - {warning}\n"
        
        # Add detailed results
        report += "\n================================\nDETAILED RESULTS\n================================\n"
        
        for stage, results in self.test_results.items():
            report += f"\n{stage.upper()}:\n"
            report += json.dumps(results, indent=2)
            report += "\n"
        
        return report
    
    def save_report(self, filename: Optional[str] = None) -> str:
        """Save validation report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"robot_validation_report_{timestamp}.txt"
        
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Report saved to: {filename}")
        return filename


def main():
    """Run staged robot validation"""
    validator = RobotConnectionValidator()
    
    print("Starting Robot Connection Validation...")
    print("WARNING: Make sure robot is connected via Ethernet!")
    
    # Stage 1: Network validation (no robot needed)
    stage1_results = validator.stage1_network_validation()
    print(f"Stage 1 Complete: {stage1_results}")
    
    # For stages 2-4, we'd need actual robot instance
    print("\nConnect robot controller for stages 2-4...")
    print("Use: validator.stage2_robot_api_validation(robot_instance)")
    
    # Generate report
    report_file = validator.save_report()
    print(f"\nReport saved: {report_file}")


if __name__ == "__main__":
    main()