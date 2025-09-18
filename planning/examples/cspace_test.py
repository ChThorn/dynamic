#!/usr/bin/env python3
"""
Configuration Space Analyzer Test

Test the C-space analyzer for reachability mapping and IK seed optimization.

Author: Robot Control Team
"""

import sys
import os
import numpy as np
import logging
import time

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'kinematics', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_configuration_space_analyzer():
    """Test C-space analyzer robustness."""
    logger.info("🧠 CONFIGURATION SPACE ANALYZER TEST")
    logger.info("=" * 50)
    
    # Import modules
    from forward_kinematic import ForwardKinematics
    from inverse_kinematic import InverseKinematics
    from configuration_space_analyzer import ConfigurationSpaceAnalyzer
    
    fk = ForwardKinematics()
    ik = InverseKinematics(fk)
    analyzer = ConfigurationSpaceAnalyzer(fk, ik)
    
    logger.info("✅ C-space analyzer initialized")
    
    # Test 1: Reachability map building
    logger.info("\n📋 Test 1: Reachability Map Building")
    logger.info("-" * 40)
    
    start_time = time.time()
    
    # Build with smaller sample sizes for testing
    analyzer.build_reachability_map(workspace_samples=100, c_space_samples=500)
    
    build_time = time.time() - start_time
    
    logger.info(f"✅ Reachability map built in {build_time:.2f}s")
    logger.info(f"   Workspace regions mapped: {len(analyzer.reachability_map)}")
    
    if len(analyzer.reachability_map) > 0:
        # Analyze map quality
        total_configs = sum(len(configs) for configs in analyzer.reachability_map.values())
        avg_configs_per_region = total_configs / len(analyzer.reachability_map)
        logger.info(f"   Total configurations: {total_configs}")
        logger.info(f"   Average configs per region: {avg_configs_per_region:.1f}")
        
        logger.info("✅ Reachability map building: PASSED")
        map_build_success = True
    else:
        logger.info("❌ No reachability regions created")
        logger.info("❌ Reachability map building: FAILED")
        map_build_success = False
    
    # Test 2: IK seed optimization
    logger.info("\n📋 Test 2: IK Seed Optimization")
    logger.info("-" * 40)
    
    if map_build_success:
        # Test targets within workspace
        test_targets = [
            np.array([0.3, 0.0, 0.6]),    # Front center
            np.array([0.2, 0.2, 0.5]),    # Right side
            np.array([0.0, 0.3, 0.7]),    # Left high
            np.array([0.4, -0.1, 0.4]),   # Front low
            np.array([-0.2, 0.0, 0.8])    # Back high
        ]
        
        seeds_found = 0
        seed_quality_scores = []
        
        for i, target in enumerate(test_targets):
            logger.info(f"   Target {i+1}: {(target*1000).round(0).astype(int)}mm")
            
            # Get IK seed from C-space analysis
            start_time = time.time()
            best_seed = analyzer.get_best_ik_region(target)
            seed_time = time.time() - start_time
            
            if best_seed is not None:
                seeds_found += 1
                logger.info(f"     ✅ Seed found in {seed_time*1000:.1f}ms: {np.rad2deg(best_seed).round(1)}°")
                
                # Test seed quality by attempting IK
                T_target = np.eye(4)
                T_target[:3, 3] = target
                
                q_solution, ik_success = ik.solve(T_target, best_seed)
                
                if ik_success:
                    # Compute TCP position to check accuracy
                    T_result = fk.compute_forward_kinematics(q_solution)
                    position_error = np.linalg.norm(T_result[:3, 3] - target)
                    
                    logger.info(f"     ✅ IK converged, error: {position_error*1000:.2f}mm")
                    seed_quality_scores.append(position_error)
                else:
                    logger.info(f"     ❌ IK failed with provided seed")
            else:
                logger.info(f"     ❌ No seed found")
        
        logger.info(f"\n   Seeds found: {seeds_found}/{len(test_targets)}")
        
        if seed_quality_scores:
            avg_error = np.mean(seed_quality_scores)
            max_error = np.max(seed_quality_scores)
            logger.info(f"   Average position error: {avg_error*1000:.2f}mm")
            logger.info(f"   Maximum position error: {max_error*1000:.2f}mm")
            
            # Quality assessment
            if avg_error < 0.005 and seeds_found >= len(test_targets) * 0.8:  # 5mm average, 80% success
                logger.info("✅ IK seed optimization: EXCELLENT")
                seed_quality = "excellent"
            elif avg_error < 0.01 and seeds_found >= len(test_targets) * 0.6:  # 10mm average, 60% success
                logger.info("✅ IK seed optimization: GOOD")
                seed_quality = "good"
            else:
                logger.info("⚠️  IK seed optimization: NEEDS IMPROVEMENT")
                seed_quality = "needs_improvement"
        else:
            logger.info("❌ IK seed optimization: FAILED")
            seed_quality = "failed"
    else:
        logger.info("❌ Cannot test IK seeds without reachability map")
        seed_quality = "failed"
    
    # Test 3: Background processing
    logger.info("\n📋 Test 3: Background Processing")
    logger.info("-" * 40)
    
    # Test async building
    async_success = analyzer.build_reachability_map_async(workspace_samples=50, c_space_samples=200)
    
    if async_success:
        logger.info("✅ Background building started")
        
        # Monitor progress
        max_wait = 10  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            progress, status = analyzer.get_build_progress()
            
            if status == "complete":
                logger.info(f"✅ Background building completed: {progress:.1f}%")
                break
            elif status == "building":
                logger.info(f"   Progress: {progress:.1f}% - {status}")
                time.sleep(1)
            else:
                break
        
        # Wait for completion
        completion_success = analyzer.wait_for_completion(timeout=15.0)
        
        if completion_success and analyzer.is_map_ready():
            logger.info("✅ Background processing: PASSED")
            background_success = True
        else:
            logger.info("❌ Background processing: TIMEOUT")
            background_success = False
    else:
        logger.info("❌ Background processing: FAILED TO START")
        background_success = False
    
    # Test 4: Caching and persistence
    logger.info("\n📋 Test 4: Caching and Persistence")
    logger.info("-" * 40)
    
    cache_dir = analyzer.cache_dir
    logger.info(f"   Cache directory: {cache_dir}")
    
    # Build a map and save it
    test_cache_path = os.path.join(cache_dir, "test_reachability_map.pkl")
    
    if map_build_success:
        try:
            analyzer._save_reachability_map(test_cache_path)
            
            if os.path.exists(test_cache_path):
                logger.info("✅ Map saved successfully")
                
                # Test loading
                original_map_size = len(analyzer.reachability_map)
                analyzer.reachability_map = {}  # Clear map
                
                load_success = analyzer.load_reachability_map(test_cache_path)
                
                if load_success and len(analyzer.reachability_map) == original_map_size:
                    logger.info("✅ Map loaded successfully")
                    logger.info("✅ Caching and persistence: PASSED")
                    caching_success = True
                else:
                    logger.info("❌ Map loading failed")
                    caching_success = False
                
                # Cleanup
                try:
                    os.remove(test_cache_path)
                except:
                    pass
            else:
                logger.info("❌ Map file not created")
                caching_success = False
        except Exception as e:
            logger.info(f"❌ Caching failed: {str(e)}")
            caching_success = False
    else:
        logger.info("❌ Cannot test caching without reachability map")
        caching_success = False
    
    # Test 5: Progress callback
    logger.info("\n📋 Test 5: Progress Callback")
    logger.info("-" * 40)
    
    progress_updates = []
    
    def test_callback(progress, status):
        progress_updates.append((progress, status))
        logger.info(f"   Callback: {progress:.1f}% - {status}")
    
    analyzer.set_progress_callback(test_callback)
    
    # Build small map to test callback
    analyzer.build_reachability_map(workspace_samples=20, c_space_samples=50)
    
    if len(progress_updates) > 0:
        logger.info(f"✅ Progress callbacks received: {len(progress_updates)}")
        
        # Check for expected progress stages
        progress_values = [update[0] for update in progress_updates]
        has_start = any(p <= 20 for p in progress_values)
        has_middle = any(20 < p < 80 for p in progress_values)
        has_end = any(p >= 90 for p in progress_values)
        
        if has_start and has_middle and has_end:
            logger.info("✅ Progress callback: PASSED")
            callback_success = True
        else:
            logger.info("⚠️  Progress callback: INCOMPLETE COVERAGE")
            callback_success = False
    else:
        logger.info("❌ Progress callback: NO CALLBACKS RECEIVED")
        callback_success = False
    
    # Summary
    logger.info("\n📊 C-SPACE ANALYZER SUMMARY")
    logger.info("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    if map_build_success:
        tests_passed += 1
        logger.info("✅ Reachability map building: PASSED")
    else:
        logger.info("❌ Reachability map building: FAILED")
    
    if seed_quality in ["excellent", "good"]:
        tests_passed += 1
        logger.info("✅ IK seed optimization: PASSED")
    else:
        logger.info("❌ IK seed optimization: FAILED")
    
    if background_success:
        tests_passed += 1
        logger.info("✅ Background processing: PASSED")
    else:
        logger.info("❌ Background processing: FAILED")
    
    if caching_success:
        tests_passed += 1
        logger.info("✅ Caching and persistence: PASSED")
    else:
        logger.info("❌ Caching and persistence: FAILED")
    
    if callback_success:
        tests_passed += 1
        logger.info("✅ Progress callback: PASSED")
    else:
        logger.info("❌ Progress callback: FAILED")
    
    success_rate = (tests_passed / total_tests) * 100
    logger.info(f"\n🎯 Overall Success Rate: {tests_passed}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("🚀 C-SPACE ANALYZER: PRODUCTION READY")
    elif success_rate >= 60:
        logger.info("⚠️  C-SPACE ANALYZER: NEEDS MINOR IMPROVEMENTS")
    else:
        logger.info("❌ C-SPACE ANALYZER: NEEDS MAJOR WORK")
    
    logger.info("\n✅ C-SPACE ANALYZER TEST COMPLETED!")

if __name__ == "__main__":
    test_configuration_space_analyzer()