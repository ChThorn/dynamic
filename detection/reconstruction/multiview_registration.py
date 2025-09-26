#!/usr/bin/env python3
"""
Multi-View Point Cloud Registration Pipeline
Fixes defective point clouds by aligning multiple views of the same object
"""

import open3d as o3d
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Dict
import copy

class PointCloudRegistration:
    """Registration pipeline for aligning multiple point cloud views"""
    
    def __init__(self, clouds_dir: Path):
        self.clouds_dir = Path(clouds_dir)
        self.registration_results = {}
        
    def load_individual_clouds(self) -> List[Tuple[str, o3d.geometry.PointCloud]]:
        """Load all individual point cloud files"""
        clouds = []
        
        # Look for individual timestamp-based PLY files (not enhanced/merged ones)
        ply_files = list(self.clouds_dir.glob("????????_??????_??????.ply"))
        
        print(f"🔍 Found {len(ply_files)} individual point cloud files")
        
        for ply_file in sorted(ply_files):
            try:
                cloud = o3d.io.read_point_cloud(str(ply_file))
                if len(cloud.points) > 100:  # Only include clouds with sufficient points
                    clouds.append((ply_file.stem, cloud))
                    print(f"   ✓ {ply_file.stem}: {len(cloud.points)} points")
                else:
                    print(f"   ✗ {ply_file.stem}: Too few points ({len(cloud.points)})")
            except Exception as e:
                print(f"   ✗ {ply_file.stem}: Load error - {e}")
        
        return clouds
    
    def preprocess_cloud(self, cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Preprocess point cloud for better registration"""
        # Remove statistical outliers
        clean_cloud, _ = cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Downsample to speed up registration
        if len(clean_cloud.points) > 10000:
            clean_cloud = clean_cloud.voxel_down_sample(voxel_size=0.002)  # 2mm voxels
        
        # Estimate normals for better registration
        clean_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
        )
        
        return clean_cloud
    
    def pairwise_icp_registration(self, 
                                source: o3d.geometry.PointCloud, 
                                target: o3d.geometry.PointCloud,
                                max_distance: float = 0.02) -> Tuple[np.ndarray, float]:
        """Perform ICP registration between two point clouds"""
        
        # Initial rough alignment using global registration
        try:
            # Use RANSAC for robust initial alignment
            result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source, target,
                o3d.pipelines.registration.compute_fpfh_feature(
                    source, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
                ),
                o3d.pipelines.registration.compute_fpfh_feature(
                    target, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=100)
                ),
                mutual_filter=True,
                max_correspondence_distance=max_distance,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                ransac_n=3,
                checkers=[
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_distance)
                ],
                criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
            )
            initial_transform = result_ransac.transformation
        except:
            # Fallback to identity if feature matching fails
            initial_transform = np.eye(4)
        
        # Refine with Point-to-Plane ICP
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target,
            max_distance,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
        )
        
        return result_icp.transformation, result_icp.fitness
    
    def register_all_clouds(self, clouds: List[Tuple[str, o3d.geometry.PointCloud]]) -> Dict:
        """Register all clouds to a common reference frame"""
        
        if len(clouds) < 2:
            print("❌ Need at least 2 point clouds for registration")
            return {}
        
        print(f"\n🔄 Registering {len(clouds)} point clouds...")
        
        # Choose reference cloud (largest one)
        reference_idx = max(range(len(clouds)), key=lambda i: len(clouds[i][1].points))
        ref_name, ref_cloud = clouds[reference_idx]
        
        print(f"📍 Reference cloud: {ref_name} ({len(ref_cloud.points)} points)")
        
        # Preprocess all clouds
        processed_clouds = []
        for name, cloud in clouds:
            processed = self.preprocess_cloud(cloud)
            processed_clouds.append((name, cloud, processed))  # Keep original + processed
        
        # Registration results
        registration_results = {
            'reference': ref_name,
            'transformations': {ref_name: np.eye(4)},  # Reference has identity transform
            'fitness_scores': {ref_name: 1.0},
            'registered_clouds': []
        }
        
        # Register each cloud to reference
        ref_processed = processed_clouds[reference_idx][2]
        
        for i, (name, original_cloud, processed_cloud) in enumerate(processed_clouds):
            if i == reference_idx:
                # Reference cloud - no transformation needed
                registered_cloud = copy.deepcopy(original_cloud)
                registration_results['registered_clouds'].append((name, registered_cloud))
                continue
            
            print(f"   🔄 Registering {name} to {ref_name}...")
            
            # Perform registration
            transformation, fitness = self.pairwise_icp_registration(
                processed_cloud, ref_processed
            )
            
            # Apply transformation to original (non-downsampled) cloud
            registered_cloud = copy.deepcopy(original_cloud)
            registered_cloud.transform(transformation)
            
            # Store results
            registration_results['transformations'][name] = transformation
            registration_results['fitness_scores'][name] = fitness
            registration_results['registered_clouds'].append((name, registered_cloud))
            
            print(f"     ✓ Fitness score: {fitness:.3f}")
            if fitness < 0.1:
                print(f"     ⚠️  Low fitness score - registration may be poor")
        
        return registration_results
    
    def merge_registered_clouds(self, registration_results: Dict) -> o3d.geometry.PointCloud:
        """Merge all registered clouds into a single point cloud"""
        
        if not registration_results or not registration_results['registered_clouds']:
            print("❌ No registered clouds to merge")
            return o3d.geometry.PointCloud()
        
        print(f"\n🔗 Merging {len(registration_results['registered_clouds'])} registered clouds...")
        
        # Combine all registered clouds
        merged_cloud = o3d.geometry.PointCloud()
        total_points = 0
        
        for name, cloud in registration_results['registered_clouds']:
            merged_cloud += cloud
            total_points += len(cloud.points)
            print(f"   + {name}: {len(cloud.points)} points")
        
        print(f"   = Total: {total_points} points")
        
        # Clean up merged cloud
        print("🧹 Cleaning merged cloud...")
        
        # Remove duplicate points
        merged_cloud = merged_cloud.remove_duplicated_points()
        print(f"   After duplicate removal: {len(merged_cloud.points)} points")
        
        # Remove statistical outliers
        merged_cloud, _ = merged_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"   After outlier removal: {len(merged_cloud.points)} points")
        
        # Optional: Uniform downsampling if too many points
        if len(merged_cloud.points) > 500000:
            merged_cloud = merged_cloud.uniform_down_sample(every_k_points=2)
            print(f"   After downsampling: {len(merged_cloud.points)} points")
        
        return merged_cloud
    
    def create_high_quality_merged_cloud(self, registration_results: Dict) -> o3d.geometry.PointCloud:
        """Create high-quality cloud using only well-registered views"""
        
        if not registration_results or not registration_results.get('fitness_scores'):
            return o3d.geometry.PointCloud()
        
        fitness_scores = registration_results['fitness_scores']
        
        print(f"\n🎯 Creating high-quality reconstruction...")
        
        # Filter clouds by registration quality
        excellent = [(name, score) for name, score in fitness_scores.items() if score > 0.9]
        good = [(name, score) for name, score in fitness_scores.items() if 0.6 < score <= 0.9]  
        poor = [(name, score) for name, score in fitness_scores.items() if 0.1 < score <= 0.6]
        bad = [(name, score) for name, score in fitness_scores.items() if score <= 0.1]
        
        print(f"   🏆 Excellent (>0.9): {len(excellent)} clouds")
        print(f"   ✅ Good (0.6-0.9): {len(good)} clouds")
        print(f"   ⚠️  Poor (0.1-0.6): {len(poor)} clouds")
        print(f"   ❌ Bad (≤0.1): {len(bad)} clouds")
        
        # Use excellent + good clouds, add poor if needed
        selected_clouds = excellent + good
        if len(selected_clouds) < 5 and poor:
            print(f"   📈 Adding {len(poor)} poor-quality clouds for better coverage")
            selected_clouds.extend(poor)
        
        print(f"   🎯 Using {len(selected_clouds)}/{len(fitness_scores)} clouds")
        
        # Find corresponding registered clouds
        registered_clouds_dict = {name: cloud for name, cloud in registration_results['registered_clouds']}
        
        # Merge selected high-quality clouds
        hq_merged = o3d.geometry.PointCloud()
        for name, score in selected_clouds:
            if name in registered_clouds_dict:
                hq_merged += registered_clouds_dict[name]
                print(f"     + {name}: {len(registered_clouds_dict[name].points)} points (fitness: {score:.3f})")
        
        # Clean the high-quality cloud more aggressively
        print(f"   🧹 Cleaning high-quality cloud...")
        hq_merged = hq_merged.remove_duplicated_points()
        hq_merged, _ = hq_merged.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        
        print(f"   ✅ Final high-quality cloud: {len(hq_merged.points)} points")
        
        return hq_merged

    def save_results(self, merged_cloud: o3d.geometry.PointCloud, 
                    registration_results: Dict, suffix: str = "registered"):
        """Save merged cloud and registration metadata"""
        
        # Save regular merged point cloud
        output_path = self.clouds_dir / f"merged_{suffix}.ply"
        o3d.io.write_point_cloud(str(output_path), merged_cloud)
        print(f"💾 Saved merged cloud: {output_path}")
        
        # Create and save high-quality version
        hq_cloud = self.create_high_quality_merged_cloud(registration_results)
        if len(hq_cloud.points) > 0:
            hq_path = self.clouds_dir / f"merged_high_quality.ply"
            o3d.io.write_point_cloud(str(hq_path), hq_cloud)
            print(f"🌟 Saved high-quality cloud: {hq_path}")
        
        # Save registration metadata
        metadata = {
            'reference_cloud': registration_results.get('reference'),
            'num_clouds': len(registration_results.get('registered_clouds', [])),
            'fitness_scores': {k: float(v) for k, v in registration_results.get('fitness_scores', {}).items()},
            'total_points': len(merged_cloud.points),
            'high_quality_points': len(hq_cloud.points) if len(hq_cloud.points) > 0 else 0,
            'method': 'ICP_with_RANSAC_initialization'
        }
        
        metadata_path = self.clouds_dir / f"registration_{suffix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📊 Saved metadata: {metadata_path}")
        
        return output_path
    
    def visualize_registration_quality(self, registration_results: Dict):
        """Visualize registration results to assess quality"""
        
        if not registration_results or not registration_results['registered_clouds']:
            print("❌ No registration results to visualize")
            return
        
        print(f"\n🎨 Visualizing registration quality...")
        
        # Color each registered cloud differently
        colors = [
            [1, 0, 0],      # Red
            [0, 1, 0],      # Green  
            [0, 0, 1],      # Blue
            [1, 1, 0],      # Yellow
            [1, 0, 1],      # Magenta
            [0, 1, 1],      # Cyan
            [0.5, 0.5, 0.5] # Gray
        ]
        
        vis_clouds = []
        for i, (name, cloud) in enumerate(registration_results['registered_clouds'][:7]):  # Max 7 clouds
            colored_cloud = copy.deepcopy(cloud)
            color = colors[i % len(colors)]
            colored_cloud.paint_uniform_color(color)
            vis_clouds.append(colored_cloud)
            print(f"   {name}: {color} ({len(cloud.points)} points)")
        
        # Visualize
        print("\n🔍 Registration Quality Visualization")
        print("   • Each cloud has a different color")
        print("   • Good registration = smooth color transitions")
        print("   • Bad registration = obvious misalignments")
        print("   • Controls: Mouse to rotate, scroll to zoom, ESC to exit")
        
        o3d.visualization.draw_geometries(
            vis_clouds,
            window_name="Point Cloud Registration Quality Check",
            width=1200,
            height=800
        )

def main():
    """Main registration pipeline"""
    
    clouds_dir = Path("detection/clouds")
    
    if not clouds_dir.exists():
        print(f"❌ Clouds directory not found: {clouds_dir}")
        return
    
    print("=== Multi-View Point Cloud Registration Pipeline ===\n")
    
    # Initialize registration system
    registrator = PointCloudRegistration(clouds_dir)
    
    # Load individual point clouds
    clouds = registrator.load_individual_clouds()
    
    if len(clouds) < 2:
        print("❌ Need at least 2 point clouds for registration")
        print("💡 Tip: Make sure you have individual timestamp-based PLY files")
        return
    
    # Perform registration
    registration_results = registrator.register_all_clouds(clouds)
    
    if not registration_results:
        print("❌ Registration failed")
        return
    
    # Merge registered clouds
    merged_cloud = registrator.merge_registered_clouds(registration_results)
    
    if len(merged_cloud.points) == 0:
        print("❌ Failed to create merged cloud")
        return
    
    # Save results
    output_path = registrator.save_results(merged_cloud, registration_results)
    
    print(f"\n🎯 Results Summary:")
    print(f"   📊 Registration Quality:")
    excellent_count = sum(1 for score in registration_results['fitness_scores'].values() if score > 0.9)
    good_count = sum(1 for score in registration_results['fitness_scores'].values() if 0.6 < score <= 0.9)
    poor_count = sum(1 for score in registration_results['fitness_scores'].values() if 0.1 < score <= 0.6)
    bad_count = sum(1 for score in registration_results['fitness_scores'].values() if score <= 0.1)
    
    print(f"     🏆 Excellent: {excellent_count}")
    print(f"     ✅ Good: {good_count}")  
    print(f"     ⚠️  Poor: {poor_count}")
    print(f"     ❌ Bad: {bad_count}")
    
    print(f"   📄 Output Files:")
    print(f"     • merged_registered.ply - All views merged")
    print(f"     • merged_high_quality.ply - Best views only")
    
    # Show individual quality scores only if requested
    print(f"   � Individual Quality Scores (fitness > 0.5 shown):")
    for name, fitness in registration_results['fitness_scores'].items():
        if fitness > 0.5:
            status = "🏆" if fitness > 0.9 else "✅" if fitness > 0.6 else "⚠️"
            print(f"     {status} {name}: {fitness:.3f}")
    
    bad_scores = [(name, score) for name, score in registration_results['fitness_scores'].items() if score <= 0.5]
    if bad_scores:
        print(f"   ❌ {len(bad_scores)} poor registrations (fitness ≤ 0.5) - excluded from high-quality version")
    
    # Visualize registration quality
    registrator.visualize_registration_quality(registration_results)
    
    print(f"\n💡 Next Steps:")
    print(f"   1. Check registration quality in the visualization")
    print(f"   2. Use merged_high_quality.ply for best results")
    print(f"   3. If still poor quality, try recapturing with fixed object method")
    print(f"   4. Quality threshold: >0.6 fitness score is good for 3D reconstruction")

if __name__ == "__main__":
    main()