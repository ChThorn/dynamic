import sys
import os
#!/usr/bin/env python3
"""
Safety-First Robot Operation Example
===================================

Demonstrates the proper safety pattern for robot initialization and mode switching.
Following the pattern from path_playing_dynamically.py with safety-first approach.

Author: Robot Control Team
Date: September 2025
"""

import time
# Add the src folder to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from planning_dynamic_executor import PlanningDynamicExecutor, create_planning_target

def demonstrate_safety_first_operation():
    """Demonstrate the safety-first robot operation pattern"""
    
    print("=" * 60)
    print("SAFETY-FIRST ROBOT OPERATION DEMONSTRATION")
    print("=" * 60)
    
    # Step 1: Initialize robot (defaults to simulation mode for safety)
    print("\n1. Initializing Robot (Safety-First Pattern)")
    print("   - Robot starts in SIMULATION mode by default")
    print("   - Real mode requires explicit activation")
    
    executor = PlanningDynamicExecutor(
        robot_ip="192.168.0.10",
        execution_mode="blend",
        chunk_size=8,
        operation_mode="simulation"  # Explicit safety mode (default)
    )
    
    # Step 2: Initialize robot components
    print("\n2. Connecting to Robot Hardware")
    if not executor.initialize():
        print("❌ Robot initialization failed")
        return False
    
    print(f"✅ Robot initialized in {executor.get_operation_mode().upper()} mode")
    
    # Step 3: Test in simulation mode (safe)
    print("\n3. Testing Motion in SIMULATION Mode (Safe)")
    target1 = create_planning_target(400, 200, 150)
    
    print("   Executing test motion in simulation...")
    success = executor.plan_and_execute_motion(target1)
    print(f"   Motion result: {'✅ Success' if success else '❌ Failed'}")
    
    # Step 4: User confirmation before real mode
    print("\n4. Switching to REAL Robot Mode")
    print("   ⚠️  WARNING: This will enable REAL PHYSICAL MOTIONS")
    print("   ⚠️  Ensure workspace is clear and safety systems are active")
    
    # In a real application, you would get user confirmation here
    user_confirmed = input("\n   Type 'ENABLE_REAL_MODE' to activate real robot: ").strip()
    
    if user_confirmed != "ENABLE_REAL_MODE":
        print("   Real mode NOT activated - staying in simulation mode")
        print("   This is the SAFE default behavior")
    else:
        # Step 5: Switch to real mode with safety validation
        print("\n   Switching to REAL mode...")
        if executor.set_operation_mode("real"):
            print(f"   ✅ Robot now in {executor.get_operation_mode().upper()} mode")
            
            # Step 6: Execute real motion (with safety warnings)
            print("\n5. Executing Motion in REAL Mode")
            print("   ⚠️  REAL PHYSICAL MOTION WILL OCCUR")
            
            target2 = create_planning_target(350, 150, 200)
            success = executor.plan_and_execute_motion(target2)
            print(f"   Motion result: {'✅ Success' if success else '❌ Failed'}")
            
        else:
            print("   ❌ Failed to switch to real mode")
    
    # Step 7: Return to safe mode
    print("\n6. Returning to Safe Mode")
    if executor.set_operation_mode("simulation"):
        print(f"   ✅ Robot returned to {executor.get_operation_mode().upper()} mode")
    
    # Step 8: Clean shutdown
    print("\n7. Safe Shutdown")
    executor.shutdown()
    print("   ✅ Robot safely shut down")
    
    print("\n" + "=" * 60)
    print("SAFETY-FIRST DEMONSTRATION COMPLETE")
    print("Key Safety Features:")
    print("✅ Default simulation mode")
    print("✅ Explicit real mode activation required")
    print("✅ Safety warnings for real operations")
    print("✅ Mode validation and logging")
    print("=" * 60)
    
    return True

def demonstrate_mode_switching():
    """Demonstrate safe mode switching patterns"""
    
    print("\n" + "=" * 40)
    print("MODE SWITCHING SAFETY PATTERNS")
    print("=" * 40)
    
    executor = PlanningDynamicExecutor()
    
    if not executor.initialize():
        print("❌ Initialization failed")
        return
    
    # Show current mode
    print(f"\nCurrent mode: {executor.get_operation_mode()}")
    print(f"Is safe mode: {executor.is_safe_mode()}")
    
    # Test invalid mode
    print("\nTesting invalid mode:")
    result = executor.set_operation_mode("invalid")
    print(f"Invalid mode result: {result}")
    
    # Test mode switching validation
    print(f"\nMode switching validation:")
    print(f"Can switch to real: {executor.set_operation_mode('real')}")
    print(f"Current mode: {executor.get_operation_mode()}")
    
    print(f"Can switch back to simulation: {executor.set_operation_mode('simulation')}")
    print(f"Current mode: {executor.get_operation_mode()}")
    
    executor.shutdown()

if __name__ == "__main__":
    try:
        print("Safety-First Robot Operation Example")
        print("Based on path_playing_dynamically.py pattern")
        
        # Run main demonstration
        demonstrate_safety_first_operation()
        
        # Run mode switching demonstration
        demonstrate_mode_switching()
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("\nExample completed")