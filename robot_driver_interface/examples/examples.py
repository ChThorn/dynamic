#!/usr/bin/env python3
"""
Planning Dynamic Executor - Example Usage
========================================

This file contains example usage of the planning dynamic executor
for position-based robot control.

Author: Robot Control Team
Date: September 2025
"""

import sys
import os


# Add the src folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planning_dynamic_executor import PlanningDynamicExecutor, create_planning_target

def basic_motion_example():
    """Basic motion control example"""
    print("Basic Motion Example")
    print("-" * 30)
    
    # Create executor in simulation mode for safety
    executor = PlanningDynamicExecutor(robot_ip="192.168.0.10", execution_mode="blend", 
                                     operation_mode="simulation")
    
    try:
        # Initialize
        if not executor.initialize():
            print("ERROR: Failed to initialize executor")
            return False
        
        print("Executor initialized successfully")
        
        # Create simple target
        target = create_planning_target(
            x_mm=400, y_mm=200, z_mm=150,
            rx_deg=180, ry_deg=0, rz_deg=0  # Pointing down
        )
        
        print(f"Moving to target: {target.tcp_position_mm}")
        
        # Execute motion
        success = executor.plan_and_execute_motion(target)
        
        if success:
            print("Motion completed successfully")
        else:
            print("ERROR: Motion failed")
        
        return success
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    finally:
        executor.shutdown()

def pick_and_place_example():
    """Pick and place sequence example"""
    print("Pick and Place Example")
    print("-" * 30)
    
    # Create executor in simulation mode for safety
    executor = PlanningDynamicExecutor(robot_ip="192.168.0.10", execution_mode="blend",
                                     operation_mode="simulation")
    
    try:
        # Initialize
        if not executor.initialize():
            print("ERROR: Failed to initialize executor")
            return False
        
        print("Executor initialized successfully")
        
        # Define pick and place locations
        pick_target = create_planning_target(
            x_mm=450, y_mm=200, z_mm=120, 
            rx_deg=180, ry_deg=0, rz_deg=0,
            gripper_mode=True  # Position is gripper tip
        )
        
        place_target = create_planning_target(
            x_mm=350, y_mm=-200, z_mm=120,
            rx_deg=180, ry_deg=0, rz_deg=0,
            gripper_mode=True  # Position is gripper tip
        )
        
        print(f"Pick location: {pick_target.tcp_position_mm}")
        print(f"Place location: {place_target.tcp_position_mm}")
        
        # Execute complete pick and place sequence
        success = executor.execute_pick_and_place_sequence(pick_target, place_target)
        
        if success:
            print("Pick and place completed successfully")
        else:
            print("ERROR: Pick and place failed")
        
        return success
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    finally:
        executor.shutdown()

def multiple_targets_example():
    """Multiple target sequence example"""
    print("Multiple Targets Example")
    print("-" * 30)
    
    # Create executor in simulation mode for safety
    executor = PlanningDynamicExecutor(robot_ip="192.168.0.10", execution_mode="blend",
                                     operation_mode="simulation")
    
    try:
        # Initialize
        if not executor.initialize():
            print("ERROR: Failed to initialize executor")
            return False
        
        print("Executor initialized successfully")
        
        # Define multiple targets
        targets = [
            create_planning_target(x_mm=300, y_mm=300, z_mm=200),
            create_planning_target(x_mm=500, y_mm=0, z_mm=250),
            create_planning_target(x_mm=300, y_mm=-300, z_mm=200),
            create_planning_target(x_mm=0, y_mm=0, z_mm=300)  # Return to center/high
        ]
        
        print(f"Planning path through {len(targets)} targets")
        
        # Execute each target
        for i, target in enumerate(targets, 1):
            print(f"  Target {i}: {target.tcp_position_mm}")
            success = executor.plan_and_execute_motion(target)
            
            if not success:
                print(f"ERROR: Failed at target {i}")
                return False
        
        print("All targets completed successfully")
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    finally:
        executor.shutdown()

def user_input_example():
    """Interactive user input example"""
    print("User Input Example")
    print("-" * 30)
    
    # Create executor in simulation mode for safety
    executor = PlanningDynamicExecutor(robot_ip="192.168.0.10", execution_mode="blend",
                                     operation_mode="simulation")
    
    try:
        # Initialize
        if not executor.initialize():
            print("ERROR: Failed to initialize executor")
            return False
        
        print("Robot ready for user commands")
        print("Enter target positions (or 'quit' to exit):")
        
        while True:
            try:
                # Get user input
                user_input = input("\nCommand (x,y,z or 'quit'): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Exiting...")
                    break
                
                # Parse coordinates
                coords = [float(x.strip()) for x in user_input.split(',')]
                
                if len(coords) != 3:
                    print("ERROR: Please enter exactly 3 coordinates: x,y,z")
                    continue
                
                x, y, z = coords
                
                # Create target
                target = create_planning_target(x_mm=x, y_mm=y, z_mm=z)
                print(f"Moving to: {target.tcp_position_mm}")
                
                # Execute motion
                success = executor.plan_and_execute_motion(target)
                
                if success:
                    print("Motion completed successfully")
                else:
                    print("ERROR: Motion failed")
                
            except ValueError:
                print("ERROR: Invalid input. Please use format: x,y,z (e.g., 400,200,150)")
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    finally:
        executor.shutdown()

def show_statistics_example():
    """Statistics and monitoring example"""
    print("Statistics Example")
    print("-" * 30)
    
    # Create executor in simulation mode for safety
    executor = PlanningDynamicExecutor(robot_ip="192.168.0.10", execution_mode="blend",
                                     operation_mode="simulation")
    
    try:
        # Initialize
        if not executor.initialize():
            print("ERROR: Failed to initialize executor")
            return False
        
        # Execute several motions to gather statistics
        targets = [
            create_planning_target(x_mm=400, y_mm=200, z_mm=150),
            create_planning_target(x_mm=300, y_mm=-200, z_mm=200),
            create_planning_target(x_mm=500, y_mm=100, z_mm=180)
        ]
        
        print("Executing motions to gather statistics...")
        
        for i, target in enumerate(targets, 1):
            print(f"  Motion {i}...")
            success = executor.plan_and_execute_motion(target)
            if not success:
                print(f"ERROR: Motion {i} failed")
        
        # Show final statistics
        stats = executor.get_statistics()
        print("\nPerformance Statistics:")
        print(f"  Total executions: {stats['total_executions']}")
        print(f"  Successful: {stats['successful_executions']}")
        print(f"  Success rate: {(stats['successful_executions']/stats['total_executions']*100):.1f}%")
        print(f"  Average planning time: {stats['average_planning_time']:.2f}s")
        print(f"  Average execution time: {stats['average_execution_time']:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False
    
    finally:
        executor.shutdown()

def main():
    """Run example demonstrations"""
    print("Planning Dynamic Executor - Examples")
    print("=" * 50)
    print("This demonstrates position-based robot control with automatic planning")
    print()
    
    # Menu
    examples = {
        '1': ('Basic Motion', basic_motion_example),
        '2': ('Pick and Place', pick_and_place_example),
        '3': ('Multiple Targets', multiple_targets_example),
        '4': ('User Input', user_input_example),
        '5': ('Statistics Demo', show_statistics_example),
        'a': ('Run All Examples', None)
    }
    
    print("Available examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print()
    
    choice = input("Select example (1-5, 'a' for all, or 'q' to quit): ").strip().lower()
    
    if choice == 'q':
        print("Goodbye!")
        return
    
    if choice == 'a':
        # Run all examples
        for key in ['1', '2', '3', '5']:  # Skip user input for auto run
            name, func = examples[key]
            print(f"\n{'='*50}")
            print(f"Running: {name}")
            print('='*50)
            func()
    elif choice in examples and choice != 'a':
        name, func = examples[choice]
        print(f"\n{'='*50}")
        print(f"Running: {name}")
        print('='*50)
        func()
    else:
        print("ERROR: Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user. Goodbye!")
    except Exception as e:
        print(f"ERROR: Unexpected error: {e}")