#!/usr/bin/env python3
"""
Quick visualization to compare before/after registration results
"""

import open3d as o3d
import numpy as np
from pathlib import Path

def compare_results():
    # Get the detection module root directory
    detection_root = Path(__file__).parent.parent
    clouds_dir = detection_root / "clouds"
    
    print("=== Point Cloud Quality Comparison ===\n")
    
    # Load different versions
    clouds = {}
    
    # Original merged (before registration)
    original_path = clouds_dir / "merged_objects.ply"
    if original_path.exists():
        clouds['original'] = o3d.io.read_point_cloud(str(original_path))
        print(f"📊 Original merged: {len(clouds['original'].points):,} points")
    
    # Full registration (all clouds)
    full_reg_path = clouds_dir / "merged_registered.ply" 
    if full_reg_path.exists():
        clouds['full_registered'] = o3d.io.read_point_cloud(str(full_reg_path))
        print(f"🔄 Full registered: {len(clouds['full_registered'].points):,} points")
    
    # High-quality registration (filtered)
    hq_path = clouds_dir / "merged_high_quality.ply"
    if hq_path.exists():
        clouds['high_quality'] = o3d.io.read_point_cloud(str(hq_path))
        print(f"✨ High-quality: {len(clouds['high_quality'].points):,} points")
    
    if not clouds:
        print("❌ No point cloud files found")
        return
    
    print(f"\n🎨 Visualizing comparison...")
    
    # Show the high-quality result (our best version)
    if 'high_quality' in clouds:
        print("🌟 Showing HIGH-QUALITY registered result")
        print("   • This should look much cleaner than before")
        print("   • Poor registrations have been filtered out")
        print("   • Object should appear more coherent and complete")
        
        # Add some color based on height for better visualization
        hq_cloud = clouds['high_quality']
        points = np.asarray(hq_cloud.points)
        
        if len(points) > 0:
            # Create height-based coloring
            z_min, z_max = points[:, 2].min(), points[:, 2].max()
            if z_max > z_min:
                normalized_z = (points[:, 2] - z_min) / (z_max - z_min)
                colors = np.zeros_like(points)
                colors[:, 0] = normalized_z      # Red increases with height
                colors[:, 1] = 1 - normalized_z  # Green decreases with height
                colors[:, 2] = 0.3               # Some blue for contrast
                hq_cloud.colors = o3d.utility.Vector3dVector(colors)
        
        # Calculate bounding box
        bbox = hq_cloud.get_axis_aligned_bounding_box()
        size = bbox.get_extent()
        print(f"   • Size: {size[0]:.3f}×{size[1]:.3f}×{size[2]:.3f}m")
        print(f"   • Controls: Mouse to rotate, scroll to zoom, ESC to exit")
        
        o3d.visualization.draw_geometries(
            [hq_cloud],
            window_name=f"HIGH-QUALITY Registered Cup ({len(points):,} points)",
            width=1200,
            height=800
        )
        
        return True
    else:
        print("❌ High-quality point cloud not found")
        return False

def main():
    success = compare_results()
    
    if success:
        print(f"\n✅ RESULT ASSESSMENT:")
        print(f"   The high-quality registered point cloud should show:")
        print(f"   🎯 Much better object coherence")
        print(f"   🧹 Reduced 'defects' and misalignments") 
        print(f"   📐 More realistic cup/mug shape")
        print(f"   🔗 Better surface continuity")
        print(f"\n💡 If it still looks defective:")
        print(f"   • The object moved too much between captures")
        print(f"   • Consider recapturing with FIXED object + moving camera")
        print(f"   • This is a fundamental limitation of the capture method")
    else:
        print(f"\n❌ Could not load results for comparison")

if __name__ == "__main__":
    main()