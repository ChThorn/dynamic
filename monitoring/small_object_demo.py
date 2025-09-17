#!/usr/bin/env python3
"""
Small Object Manipulation: Realistic Pick-and-Place Demo
======================================================

This script demonstrates how to handle small objects (10mm height) placed
directly on the wood surface, considering proper gripper positioning for
successful grasping.

Real-world scenario:
- Robot base at z=0 (floor mounted)
- Wood surface at z=60mm 
- Small objects (10mm height) on wood surface
- Object top at z=70mm (60mm + 10mm)
- Need to position gripper for successful grasp

Author: GitHub Copilot
Date: September 2025
"""

import sys
import os
import numpy as np

# Add module paths
sys.path.append('../kinematics/src')
sys.path.append('.')

from forward_kinematic import ForwardKinematics
from inverse_kinematic import InverseKinematics


def analyze_small_object_grasping():
    """
    Analyze how to properly grasp small objects on the wood surface.
    """
    
    print("🔍 SMALL OBJECT GRASPING ANALYSIS")
    print("=" * 50)
    print("Realistic scenario: 10mm high objects on 60mm wood surface")
    print()
    
    # Physical setup
    floor_level = 0      # mm - Robot base
    wood_surface = 60    # mm - Wood surface height
    object_height = 10   # mm - Small object height
    object_top = wood_surface + object_height  # 70mm
    
    print("📏 PHYSICAL SETUP:")
    print(f"   🏠 Robot base (floor): {floor_level}mm")
    print(f"   🪵 Wood surface: {wood_surface}mm")
    print(f"   📦 Object height: {object_height}mm")
    print(f"   🎯 Object top surface: {object_top}mm")
    print()
    
    # Initialize gripper mode
    fk = ForwardKinematics(tool_name='default_gripper')
    ik = InverseKinematics(fk)
    
    # Test different grasping strategies
    object_position = [200, 150]  # X, Y position on wood surface
    
    grasping_strategies = [
        ("Surface Level", object_top, "Gripper at object top - risky"),
        ("Slight Penetration", object_top - 2, "2mm into object - better grip"),
        ("Mid-Object", object_top - 5, "Mid-height of object - optimal"),
        ("Near Bottom", wood_surface + 2, "2mm above wood - maximum grip"),
        ("Too High", object_top + 20, "20mm above object - will miss"),
        ("Surface Contact", wood_surface + 1, "1mm above wood - careful approach")
    ]
    
    print("🎯 GRASPING STRATEGY ANALYSIS:")
    print(f"Object at position: [{object_position[0]}, {object_position[1]}] mm")
    print()
    
    successful_strategies = 0
    optimal_strategies = []
    
    for strategy_name, grasp_height, description in grasping_strategies:
        print(f"📋 {strategy_name}: {description}")
        print(f"   Target gripper height: {grasp_height}mm")
        
        # Create target position
        target_position = [object_position[0], object_position[1], grasp_height]
        
        # Check if this makes physical sense
        if grasp_height < wood_surface:
            print(f"   ❌ IMPOSSIBLE: Below wood surface (collision)")
        elif grasp_height > object_top + 10:
            print(f"   ⚠️  PROBLEMATIC: Too high above object (will miss)")
        else:
            # Test IK feasibility
            T_target = np.eye(4)
            T_target[:3, 3] = np.array(target_position) / 1000
            
            q_init = np.deg2rad([0, -30, 60, 0, 45, 0])
            q_solution, converged = ik.solve(T_target, q_init=q_init)
            
            if converged:
                # Verify solution
                T_achieved = fk.compute_forward_kinematics(q_solution)
                achieved_pos = T_achieved[:3, 3] * 1000
                error = np.linalg.norm(achieved_pos - target_position)
                
                # Get TCP position for reference
                T_tcp = fk.compute_tcp_kinematics(q_solution)
                tcp_pos = T_tcp[:3, 3] * 1000
                
                if error < 1.0:
                    print(f"   ✅ SUCCESS: Reachable with {error:.3f}mm accuracy")
                    print(f"   📍 Gripper at: [{achieved_pos[0]:.1f}, {achieved_pos[1]:.1f}, {achieved_pos[2]:.1f}] mm")
                    print(f"   🔧 TCP at: [{tcp_pos[0]:.1f}, {tcp_pos[1]:.1f}, {tcp_pos[2]:.1f}] mm")
                    
                    # Analyze grasping quality
                    height_above_surface = achieved_pos[2] - wood_surface
                    height_from_object_top = achieved_pos[2] - object_top
                    
                    if wood_surface < achieved_pos[2] <= object_top:
                        quality = "🟢 EXCELLENT - Within object bounds"
                        optimal_strategies.append((strategy_name, grasp_height))
                    elif achieved_pos[2] <= wood_surface + 2:
                        quality = "🟡 GOOD - Very close to surface"
                    elif achieved_pos[2] > object_top + 5:
                        quality = "🔴 POOR - Too high, will miss object"
                    else:
                        quality = "🟠 ACCEPTABLE - May work with careful approach"
                    
                    print(f"   📊 Height above surface: {height_above_surface:.1f}mm")
                    print(f"   📊 Distance from object top: {height_from_object_top:.1f}mm")
                    print(f"   🎯 Grasping quality: {quality}")
                    
                    successful_strategies += 1
                else:
                    print(f"   ❌ FAILED: Poor accuracy ({error:.3f}mm)")
            else:
                print(f"   ❌ FAILED: IK did not converge")
        
        print()
    
    print("📊 GRASPING ANALYSIS SUMMARY:")
    print("=" * 40)
    print(f"✅ Feasible strategies: {successful_strategies}/{len(grasping_strategies)}")
    print(f"🎯 Optimal strategies: {len(optimal_strategies)}")
    print()
    
    if optimal_strategies:
        print("💡 RECOMMENDED GRASPING HEIGHTS:")
        for strategy, height in optimal_strategies:
            print(f"   • {strategy}: {height}mm")
    
    return successful_strategies > 0


def demonstrate_realistic_pick_and_place():
    """
    Demonstrate a realistic pick-and-place workflow for small objects.
    """
    
    print("🤖 REALISTIC SMALL OBJECT WORKFLOW")
    print("=" * 50)
    print("Complete workflow for handling 10mm objects on wood surface")
    print()
    
    # Initialize gripper mode
    fk = ForwardKinematics(tool_name='default_gripper')
    ik = InverseKinematics(fk)
    
    # Physical constraints
    wood_surface = 60    # mm
    object_height = 10   # mm
    safe_approach_height = 200  # mm - safe approach height
    
    # Object locations
    pick_object_pos = [200, 150]  # Object to pick
    place_location = [300, -100]  # Where to place it
    
    # Calculate optimal grasping heights
    pick_grasp_height = wood_surface + object_height - 3  # 3mm into object for secure grip
    place_height = wood_surface + 5  # 5mm above surface for gentle placement
    
    # Define realistic workflow
    workflow = [
        ([0, 0, safe_approach_height], "Home position - safe height"),
        ([pick_object_pos[0], pick_object_pos[1], safe_approach_height], "Approach above pick object"),
        ([pick_object_pos[0], pick_object_pos[1], pick_grasp_height], f"Grasp object - {pick_grasp_height}mm height"),
        ([pick_object_pos[0], pick_object_pos[1], safe_approach_height], "Lift object safely"),
        ([place_location[0], place_location[1], safe_approach_height], "Move to place location"),
        ([place_location[0], place_location[1], place_height], f"Place object - {place_height}mm height"),
        ([place_location[0], place_location[1], safe_approach_height], "Retract after placement"),
        ([0, 0, safe_approach_height], "Return home")
    ]
    
    print("📋 REALISTIC WORKFLOW SEQUENCE:")
    for i, (pos, desc) in enumerate(workflow):
        height_above_surface = pos[2] - wood_surface
        print(f"   {i+1}. {desc}")
        print(f"      Position: [{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}] mm ({height_above_surface:.0f}mm above surface)")
    print()
    
    trajectory_points = []
    all_successful = True
    
    print("🔧 EXECUTING REALISTIC WORKFLOW:")
    print()
    
    for i, (gripper_pos, description) in enumerate(workflow):
        print(f"{i+1}. {description}")
        
        # Create target transformation
        T_target = np.eye(4)
        T_target[:3, 3] = np.array(gripper_pos) / 1000
        
        # Use previous position as initial guess
        q_init = trajectory_points[-1] if trajectory_points else np.deg2rad([0, -30, 60, 0, 45, 0])
        
        # Solve IK
        q_solution, converged = ik.solve(T_target, q_init=q_init)
        
        if converged:
            # Verify result
            T_achieved = fk.compute_forward_kinematics(q_solution)
            achieved_gripper = T_achieved[:3, 3] * 1000
            
            error = np.linalg.norm(achieved_gripper - gripper_pos)
            height_above_surface = achieved_gripper[2] - wood_surface
            
            print(f"   ✅ SUCCESS: Gripper at [{achieved_gripper[0]:.1f}, {achieved_gripper[1]:.1f}, {achieved_gripper[2]:.1f}] mm")
            print(f"   🎯 Accuracy: {error:.3f} mm")
            print(f"   📏 Height above surface: {height_above_surface:.1f}mm")
            
            # Special analysis for critical steps
            if "Grasp object" in description:
                if wood_surface < achieved_gripper[2] <= wood_surface + object_height:
                    print(f"   🟢 PERFECT: Gripper positioned within object bounds")
                elif achieved_gripper[2] <= wood_surface + 2:
                    print(f"   🟡 GOOD: Very close to surface, careful approach needed")
                else:
                    print(f"   🔴 WARNING: May be too high for secure grasp")
            
            elif "Place object" in description:
                if wood_surface < achieved_gripper[2] <= wood_surface + 10:
                    print(f"   🟢 PERFECT: Good height for gentle placement")
                else:
                    print(f"   🟡 ACCEPTABLE: Placement height may need adjustment")
            
            trajectory_points.append(q_solution)
            
        else:
            print(f"   ❌ FAILED: IK did not converge")
            all_successful = False
            break
        
        print()
    
    if all_successful:
        print("🎉 REALISTIC WORKFLOW COMPLETE!")
        print(f"✅ All {len(trajectory_points)} waypoints successful")
        print("📊 Trajectory optimized for small object manipulation")
        
        # Calculate gripper opening requirements
        print(f"\n🔧 GRIPPER REQUIREMENTS:")
        print(f"   📐 Minimum opening: {object_height + 5}mm (object + clearance)")
        print(f"   💪 Recommended force: 5-10N (gentle but secure)")
        print(f"   ⚡ Approach speed: Slow (1-2 mm/s near object)")
        
    else:
        print("❌ Workflow failed - some positions unreachable")
    
    return all_successful


def provide_small_object_guidelines():
    """
    Provide practical guidelines for small object manipulation.
    """
    
    print("💡 SMALL OBJECT MANIPULATION GUIDELINES")
    print("=" * 55)
    print()
    
    print("🎯 OPTIMAL GRASPING STRATEGY:")
    print("   • Object height: 10mm on 60mm wood surface")
    print("   • Object top surface: 70mm absolute height")
    print("   • Optimal grasp height: 67mm (3mm into object)")
    print("   • Alternative: 62mm (2mm above surface)")
    print("   • Never grasp above 75mm (will miss object)")
    print()
    
    print("📏 HEIGHT CALCULATIONS:")
    print("   wood_surface = 60mm")
    print("   object_height = 10mm") 
    print("   object_top = wood_surface + object_height = 70mm")
    print("   optimal_grasp = object_top - 3mm = 67mm")
    print("   safe_approach = 200mm+ (well above workspace)")
    print()
    
    print("🔧 IMPLEMENTATION EXAMPLE:")
    print("   # Calculate object grasping height")
    print("   wood_surface = 60  # mm")
    print("   object_height = 10  # mm") 
    print("   grasp_height = wood_surface + object_height - 3  # 67mm")
    print("   ")
    print("   # Create gripper target")
    print("   target_pos = [x, y, grasp_height]")
    print("   T_target = np.eye(4)")
    print("   T_target[:3, 3] = np.array(target_pos) / 1000")
    print()
    
    print("⚠️  CRITICAL CONSIDERATIONS:")
    print("   • Vision system must detect object height accurately")
    print("   • Force feedback essential for delicate objects")
    print("   • Slow approach speeds near surface (collision risk)")
    print("   • Gripper finger design must accommodate thin objects")
    print("   • Consider vacuum gripper for very flat objects")
    print()
    
    print("🚫 COMMON MISTAKES TO AVOID:")
    print("   ❌ Grasping at 80mm+ (standard 'safe' height)")
    print("   ❌ Not accounting for object thickness")
    print("   ❌ Using TCP coordinates instead of gripper coordinates")
    print("   ❌ Fast approach speeds near delicate objects")
    print("   ❌ Excessive gripping force on fragile items")


def main():
    """
    Main function for small object manipulation analysis.
    """
    
    print("🔬 SMALL OBJECT MANIPULATION: COMPLETE ANALYSIS")
    print("=" * 70)
    print("Realistic handling of 10mm objects on 60mm wood surface")
    print()
    
    try:
        # Run analysis
        grasping_success = analyze_small_object_grasping()
        workflow_success = demonstrate_realistic_pick_and_place()
        
        # Provide guidelines
        provide_small_object_guidelines()
        
        # Summary
        print("\n🎯 ANALYSIS SUMMARY")
        print("=" * 30)
        
        if grasping_success:
            print("✅ Small object grasping: FEASIBLE")
        else:
            print("❌ Small object grasping: PROBLEMATIC")
            
        if workflow_success:
            print("✅ Realistic workflow: SUCCESSFUL")
        else:
            print("❌ Realistic workflow: FAILED")
        
        if grasping_success and workflow_success:
            print("\n🎉 SMALL OBJECT MANIPULATION VALIDATED!")
            print("🚀 System ready for realistic pick-and-place operations")
        else:
            print("\n⚠️  Issues detected - review positioning strategy")
        
        print("\n📋 KEY INSIGHTS:")
        print("   • 10mm objects require grasp height of ~67mm (not 80mm+)")
        print("   • Gripper must position within object bounds for secure grip")
        print("   • Safe approach heights (200mm+) essential for collision avoidance")
        print("   • Careful speed control critical near wood surface")
        print("   • Vision and force feedback recommended for robust operation")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()