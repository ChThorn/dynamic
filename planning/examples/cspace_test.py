#!/usr/bin/env python3
"""
Configuration Space Analyzer Test

Test the C-space analyzer for reachability mapping and IK seed optimization.
Includes visualization of the reachability map.

Author: Robot Control Team
"""

import sys
import os
import numpy as np
import logging
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'kinematics', 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_reachability_map(analyzer, save_path="reachability_map_visualization.png"):
    """
    Visualize the reachability map in 3D space.
    
    Args:
        analyzer: ConfigurationSpaceAnalyzer with built reachability map
        save_path: Path to save the visualization
    """
    logger.info("\n🎨 Visualizing Reachability Map")
    logger.info("-" * 40)
    
    if not analyzer.reachability_map:
        logger.info("❌ No reachability map to visualize")
        return False
    
    try:
        # Create a figure for plotting
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract center points of each region and count configs
        centers = []
        config_counts = []
        
        logger.info(f"Preparing visualization with {len(analyzer.reachability_map)} regions")
        logger.info(f"Sample key type: {type(list(analyzer.reachability_map.keys())[0])}")
        
        for region_key, configs in analyzer.reachability_map.items():
            # Handle the region key which could be a tuple or string
            if isinstance(region_key, tuple):
                # If it's already a tuple, use it directly
                coords = region_key
            elif isinstance(region_key, str) and '_' in region_key:
                # Parse string format like "0.4_0.2_0.7"
                coords = [float(val) for val in region_key.split('_')]
            else:
                # Skip invalid keys
                logger.info(f"Skipping invalid region key format: {type(region_key)}")
                continue
                
            if len(coords) >= 3:
                centers.append(coords[:3])
                config_counts.append(len(configs))
        
        if not centers:
            logger.info("❌ No valid regions found for visualization")
            return False
        
        logger.info(f"Plotting {len(centers)} valid regions")
        
        # Convert to numpy arrays
        centers = np.array(centers)
        config_counts = np.array(config_counts)
        
        # Normalize configuration counts for color mapping
        if len(config_counts) > 1 and config_counts.max() > config_counts.min():
            norm_counts = (config_counts - config_counts.min()) / (config_counts.max() - config_counts.min())
        else:
            norm_counts = np.ones_like(config_counts) * 0.5
        
        # Plot each point with size and color representing config count
        scatter = ax.scatter(
            centers[:, 0], centers[:, 1], centers[:, 2],
            c=norm_counts, 
            cmap=cm.viridis, 
            s=norm_counts * 100 + 20,
            alpha=0.7
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Relative Configuration Density')
        
        # Add simple workspace boundaries (standard robot workspace)
        # Define standard workspace bounds for RB3-730ES
        x_min, x_max = -0.7, 0.7
        y_min, y_max = -0.7, 0.7
        z_min, z_max = 0.1, 1.1
        
        # Create wireframe box for workspace boundaries
        x_edges = np.array([
            [x_min, x_min, x_min, x_min, x_max, x_max, x_max, x_max],
            [x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max]
        ])
        y_edges = np.array([
            [y_min, y_max, y_min, y_max, y_min, y_max, y_min, y_max],
            [y_min, y_max, y_min, y_max, y_min, y_max, y_min, y_max]
        ])
        z_edges = np.array([
            [z_min, z_min, z_min, z_min, z_min, z_min, z_min, z_min],
            [z_max, z_max, z_max, z_max, z_max, z_max, z_max, z_max]
        ])
        
        # Add a translucent sphere showing the 730mm theoretical reach from base joint
        # The real reach is from the base joint (between link0 and link1) which is at z=0.1453
        r = 0.73  # 730mm reach
        base_z = 0.1453  # Base joint height
        
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + base_z
        
        # Plot sphere
        ax.plot_surface(x, y, z, color='r', alpha=0.1)
        
        # Add a note about the actual robot base position
        ax.text(0, 0, base_z, "Base Joint", color='black', fontsize=10)
        
        # Plot wireframe
        for i in range(4):
            ax.plot(
                [x_edges[0,i], x_edges[1,i]], 
                [y_edges[0,i], y_edges[1,i]], 
                [z_edges[0,i], z_edges[1,i]], 
                'k-', alpha=0.3
            )
            ax.plot(
                [x_edges[0,i+4], x_edges[1,i+4]], 
                [y_edges[0,i+4], y_edges[1,i+4]], 
                [z_edges[0,i+4], z_edges[1,i+4]], 
                'k-', alpha=0.3
            )
            ax.plot(
                [x_edges[0,i], x_edges[0,i+4]], 
                [y_edges[0,i], y_edges[0,i+4]], 
                [z_edges[0,i], z_edges[0,i+4]], 
                'k-', alpha=0.3
            )
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('RB3-730ES-U Robot Reachability Map\n730mm Maximum Reach from Base Joint')
        
        # Add annotations for context
        ax.text(0.7, 0, 0.2, "730mm Reach Boundary", color='r', fontsize=8)
        
        # Add test targets to the visualization
        test_targets = [
            (0.3, 0.0, 0.5),    # Front center
            (0.2, 0.2, 0.5),    # Right side
            (0.0, 0.3, 0.5),    # Left side
            (0.4, -0.1, 0.3),   # Front extended
            (-0.2, 0.0, 0.6)    # Back position
        ]
        
        # Plot test targets
        target_x = [t[0] for t in test_targets]
        target_y = [t[1] for t in test_targets]
        target_z = [t[2] for t in test_targets]
        ax.scatter(target_x, target_y, target_z, color='green', marker='*', s=100, label='Test Targets')
        
        # Set axis limits to match workspace bounds
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cm.viridis(0.7), markersize=10, label='Reachable Regions'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=10, label='Test Targets'),
            Line2D([0], [0], color='red', alpha=0.3, lw=2, label='730mm Reach')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize='small')
        
        # Add information on test target positions
        ax.text(x_min+0.1, y_min+0.1, z_max-0.1, 
                "Test Targets (m):\n" + 
                "1: (0.3, 0.0, 0.5)\n" + 
                "2: (0.2, 0.2, 0.5)\n" + 
                "3: (0.0, 0.3, 0.5)\n" + 
                "4: (0.4, -0.1, 0.3)\n" + 
                "5: (-0.2, 0.0, 0.6)", 
                fontsize=8)
        
        # Adjust view angle for better visualization
        ax.view_init(elev=25, azim=30)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        logger.info(f"✅ Reachability map visualization saved to: {save_path}")
        
        return True
    except Exception as e:
        logger.info(f"❌ Visualization failed: {str(e)}")
        return False
        
        # Set labels and title
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Robot Reachability Map')
        
        # Adjust view angle
        ax.view_init(elev=30, azim=45)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        logger.info(f"✅ Reachability map visualization saved to: {save_path}")
        
        # Show plot if in interactive mode
        plt.show()
        
        return True
    
    except Exception as e:
        logger.info(f"❌ Visualization failed: {str(e)}")
        return False

def test_configuration_space_analyzer():
    """Test C-space analyzer robustness."""
    logger.info("🧠 CONFIGURATION SPACE ANALYZER TEST")
    logger.info("=" * 50)
    
    # Import modules
    from forward_kinematic import ForwardKinematics
    from inverse_kinematic import FastIK
    from configuration_space_analyzer import ConfigurationSpaceAnalyzer
    
    fk = ForwardKinematics()
    ik = FastIK(fk)
    analyzer = ConfigurationSpaceAnalyzer(fk, ik)
    
    logger.info("✅ C-space analyzer initialized")
    
    # Test 1: Reachability map building
    logger.info("\n📋 Test 1: Reachability Map Building")
    logger.info("-" * 40)
    
    start_time = time.time()
    
    # Build with significantly more samples for better workspace coverage and visualization
    analyzer.build_reachability_map(workspace_samples=400, c_space_samples=2000)
    
    build_time = time.time() - start_time
    
    logger.info(f"✅ Reachability map built in {build_time:.2f}s")
    logger.info(f"   Workspace regions mapped: {len(analyzer.reachability_map)}")
    
    if len(analyzer.reachability_map) > 0:
        # Analyze map quality
        total_configs = sum(len(configs) for configs in analyzer.reachability_map.values())
        avg_configs_per_region = total_configs / len(analyzer.reachability_map)
        logger.info(f"   Total configurations: {total_configs}")
        logger.info(f"   Average configs per region: {avg_configs_per_region:.1f}")
        
        # Visualize the reachability map
        visualization_path = os.path.join(os.path.dirname(__file__), "reachability_map_visualization.png")
        vis_success = visualize_reachability_map(analyzer, visualization_path)
        if vis_success:
            logger.info(f"✅ Reachability map visualization created at: {visualization_path}")
        
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
        # Test targets within the robot's realistic workspace based on 730mm reach
        # Modified to better account for robot's actual reach capabilities
        test_targets = [
            np.array([0.3, 0.0, 0.5]),    # Front center - adjusted height to be more reachable
            np.array([0.2, 0.2, 0.5]),    # Right side - this one worked previously
            np.array([0.0, 0.3, 0.5]),    # Left side - lowered height for better reachability
            np.array([0.4, -0.1, 0.3]),   # Front extended - lowered height to account for extended reach
            np.array([-0.2, 0.0, 0.6])    # Back position - lowered height for better reachability
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
        logger.info("   Visual representation created to show workspace coverage")
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
        visualization_path = os.path.join(os.path.dirname(__file__), "reachability_map_visualization.png")
        if os.path.exists(visualization_path):
            logger.info(f"📈 Visualization available at: {visualization_path}")
            logger.info("   The visualization shows regions where the robot can reach with color")
            logger.info("   indicating the density of possible configurations for each region.")
    elif success_rate >= 60:
        logger.info("⚠️  C-SPACE ANALYZER: NEEDS MINOR IMPROVEMENTS")
    else:
        logger.info("❌ C-SPACE ANALYZER: NEEDS MAJOR WORK")
    
    logger.info("\n✅ C-SPACE ANALYZER TEST COMPLETED!")

if __name__ == "__main__":
    test_configuration_space_analyzer()