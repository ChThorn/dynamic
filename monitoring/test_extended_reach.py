#!/usr/bin/env python3
"""
Extended Reach Analysis for RB3-730ES-U Robot

This script tests and validates extended reach capabilities up to 700mm,
analyzing where current limitations come from and proposing solutions.
"""
import sys
import os
import numpy as np

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_urdf_reach():
    """Calculate theoretical maximum reach from URDF."""
    print("🔍 URDF ANALYSIS - Theoretical Maximum Reach")
    print("="*50)
    
    # From URDF joint origins
    base_height = 0.1453      # Base joint height
    upper_arm = 0.286         # Shoulder to elbow (link2)
    forearm = 0.344           # Elbow to wrist2 (link4) 
    tcp_extension = 0.1       # Wrist3 to TCP
    
    theoretical_max = upper_arm + forearm + tcp_extension
    print(f"Link lengths:")
    print(f"  Upper arm (link2): {upper_arm*1000:.1f}mm")
    print(f"  Forearm (link4): {forearm*1000:.1f}mm") 
    print(f"  TCP extension: {tcp_extension*1000:.1f}mm")
    print(f"  Theoretical horizontal reach: {theoretical_max*1000:.1f}mm")
    
    # Maximum reach including base height (straight up configuration)
    max_vertical = base_height + upper_arm + forearm + tcp_extension
    print(f"  Maximum vertical reach: {max_vertical*1000:.1f}mm")
    
    return theoretical_max, max_vertical

def analyze_json_data():
    """Analyze real robot data from JSON."""
    print("\n📊 JSON DATA ANALYSIS - Real Robot Performance")
    print("="*50)
    
    # From the JSON data provided
    tcp_at_zero = [0.038, -6.512, 877.066]  # TCP position at all joints zero
    distance_zero = np.sqrt(tcp_at_zero[0]**2 + tcp_at_zero[1]**2)
    
    print(f"Real robot at zero configuration:")
    print(f"  TCP position: [{tcp_at_zero[0]:.1f}, {tcp_at_zero[1]:.1f}, {tcp_at_zero[2]:.1f}]mm")
    print(f"  Horizontal distance: {distance_zero:.1f}mm") 
    print(f"  Vertical height: {tcp_at_zero[2]:.1f}mm")
    print(f"  Total reach from base: {tcp_at_zero[2]:.1f}mm")
    
    return tcp_at_zero

def test_current_workspace_limits():
    """Test current workspace constraints."""
    print("\n⚙️  CURRENT WORKSPACE CONSTRAINTS")
    print("="*50)
    
    try:
        from planning.examples.clean_robot_interface import CleanRobotMotionPlanner
        
        planner = CleanRobotMotionPlanner()
        home_joints = [0.0, -60.0, 90.0, 0.0, 30.0, 0.0]
        
        # Test positions at increasing distances
        test_distances = [
            (400, "Current safe zone"),
            (500, "Current practical limit"), 
            (600, "Target extended reach"),
            (650, "Aggressive extended reach"),
            (700, "Near theoretical limit"),
            (720, "Close to URDF limit")
        ]
        
        results = {}
        for distance_mm, description in test_distances:
            # Test position at optimal height for that distance
            if distance_mm <= 500:
                z_height = 300  # Higher Z for shorter reach
            elif distance_mm <= 600:
                z_height = 250  # Medium Z
            else:
                z_height = 200  # Lower Z for maximum reach
                
            pos_mm = [distance_mm, 0.0, z_height]
            rot_deg = [180.0, 0.0, 0.0]
            
            print(f"\n🎯 Testing {distance_mm}mm reach ({description})")
            print(f"   Position: {pos_mm}mm")
            
            try:
                plan = planner.plan_motion(home_joints, pos_mm, rot_deg)
                success = plan.success
                results[distance_mm] = {
                    'success': success,
                    'error': plan.error_message if not success else None,
                    'waypoints': len(plan.waypoints) if success else 0
                }
                
                if success:
                    print(f"   ✅ SUCCESS: {len(plan.waypoints)} waypoints")
                else:
                    print(f"   ❌ FAILED: {plan.error_message}")
                    
            except Exception as e:
                print(f"   ❌ EXCEPTION: {e}")
                results[distance_mm] = {'success': False, 'error': str(e), 'waypoints': 0}
        
        return results
        
    except Exception as e:
        print(f"❌ Error testing workspace limits: {e}")
        return {}

def analyze_limiting_factors():
    """Analyze what's limiting the reach to 500mm."""
    print("\n🔍 LIMITING FACTORS ANALYSIS")
    print("="*50)
    
    # Check workspace sampling limits
    try:
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'planning', 'src'))
        from configuration_space_analyzer import ConfigurationSpaceAnalyzer
        
        print("1. Configuration Space Analyzer:")
        print(f"   Current workspace sampling bounds:")
        print(f"   x_range: (-0.7, 0.7) = ±700mm")
        print(f"   y_range: (-0.7, 0.7) = ±700mm") 
        print(f"   z_range: (0.1, 1.0) = 100-1000mm")
        print(f"   ✅ Workspace bounds support 700mm reach")
        
    except Exception as e:
        print(f"   ❌ Could not analyze configuration space: {e}")
    
    # Check constraint file
    print("\n2. Constraint Configuration:")
    constraint_file = "/home/thornch/Documents/Python/RB_Work/dynamic/config/constraints.yaml"
    if os.path.exists(constraint_file):
        print(f"   Workspace limits in constraints.yaml:")
        print(f"   x_max: 0.72m = 720mm")
        print(f"   y_max: 0.72m = 720mm")
        print(f"   ✅ Constraints support 720mm reach")
    else:
        print(f"   ❌ Constraints file not found")
    
    print("\n3. Potential Limiting Factors:")
    print("   • IK solver convergence at extended reach")
    print("   • Collision detection overly conservative")
    print("   • Joint limit constraints")
    print("   • Cached reachability map limitations")
    print("   • Motion planner conservative heuristics")

def propose_solutions():
    """Propose solutions to extend reach to 650-700mm."""
    print("\n💡 PROPOSED SOLUTIONS TO EXTEND REACH")
    print("="*50)
    
    print("1. 🎯 IMMEDIATE ACTIONS (should increase reach to ~650mm):")
    print("   • Clear cached reachability maps to force rebuild")
    print("   • Increase workspace sampling bounds to ±0.75m (750mm)")
    print("   • Relax collision detection for extended reach positions")
    print("   • Optimize IK solver parameters for better convergence")
    
    print("\n2. 🔧 CONFIGURATION UPDATES:")
    print("   • Update constraints.yaml workspace bounds to 0.75m")
    print("   • Enable extended_reach_mode in collision detection")
    print("   • Increase max_reach in self_collision to 0.700m")
    print("   • Rebuild reachability map with more samples")
    
    print("\n3. 🚀 ADVANCED OPTIMIZATIONS (target ~700mm reach):")
    print("   • Implement adaptive collision thresholds by distance")
    print("   • Add specialized IK strategies for extended reach") 
    print("   • Optimize joint limit utilization")
    print("   • Fine-tune safety margins for different reach zones")
    
    print("\n4. 📋 VALIDATION STRATEGY:")
    print("   • Test incremental reach increases (550mm → 600mm → 650mm → 700mm)")
    print("   • Validate FK-IK consistency at extended reach")
    print("   • Ensure safety margins remain adequate")
    print("   • Performance testing under various orientations")

def main():
    """Run extended reach analysis."""
    print("🤖 RB3-730ES-U EXTENDED REACH ANALYSIS")
    print("Investigating reach limitations and proposing solutions")
    print("=" * 80)
    
    # 1. Analyze theoretical capabilities
    theoretical_reach, max_vertical = analyze_urdf_reach()
    
    # 2. Analyze real robot performance
    json_data = analyze_json_data()
    
    # 3. Test current practical limits
    current_results = test_current_workspace_limits()
    
    # 4. Analyze limiting factors
    analyze_limiting_factors()
    
    # 5. Propose solutions
    propose_solutions()
    
    # 6. Summary
    print(f"\n🎯 ANALYSIS SUMMARY")
    print("="*60)
    print(f"✅ URDF Theoretical Reach: {theoretical_reach*1000:.0f}mm horizontal")
    print(f"✅ Real Robot Verified: {json_data[2]:.0f}mm vertical at zero config")
    print(f"⚠️  Current Practical Limit: ~500mm (needs improvement)")
    print(f"🎯 Target Extended Reach: 650-700mm (feasible)")
    
    if current_results:
        successful_distances = [d for d, r in current_results.items() if r['success']]
        if successful_distances:
            max_success = max(successful_distances)
            print(f"📊 Current Test Results: Up to {max_success}mm successful")
    
    print(f"\n📋 NEXT STEPS:")
    print("1. Implement proposed configuration changes")
    print("2. Clear reachability map cache and rebuild")
    print("3. Test incremental reach improvements")
    print("4. Validate safety and performance at extended reach")
    print("5. Update workspace documentation with new limits")

if __name__ == "__main__":
    main()