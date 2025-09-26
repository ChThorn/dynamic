#!/usr/bin/env python3
"""
3D Point Cloud Generator for captured RGB-D + mask data.

This script generates 3D point clouds from the captured data using:
- RGB images for color
- Depth images for 3D coordinates  
- Segmentation masks for object filtering
- Camera intrinsics for proper projection
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d


class PointCloudGenerator:
    """Generate 3D point clouds from captured RGB-D data."""
    
    def __init__(self, captures_dir: Path):
        """
        Initialize point cloud generator.
        
        Args:
            captures_dir: Directory containing captured RGB/depth/mask/meta data
        """
        self.captures_dir = Path(captures_dir)
        self.rgb_dir = self.captures_dir / "rgb"
        self.depth_dir = self.captures_dir / "depth"
        self.mask_dir = self.captures_dir / "mask"
        self.meta_dir = self.captures_dir / "meta"
        
        # Output directory for point clouds
        self.clouds_dir = self.captures_dir.parent / "clouds"
        self.clouds_dir.mkdir(exist_ok=True)
    
    def load_frame_data(self, frame_id: str) -> Optional[dict]:
        """Load all data for a specific frame."""
        rgb_path = self.rgb_dir / f"{frame_id}.png"
        depth_path = self.depth_dir / f"{frame_id}.png"
        mask_path = self.mask_dir / f"{frame_id}.png"
        meta_path = self.meta_dir / f"{frame_id}.json"
        
        # Check if all files exist
        if not all(p.exists() for p in [rgb_path, depth_path, mask_path, meta_path]):
            missing = [p.name for p in [rgb_path, depth_path, mask_path, meta_path] if not p.exists()]
            print(f"⚠️  Missing files for {frame_id}: {missing}")
            return None
        
        # Load images
        rgb_image = cv2.imread(str(rgb_path))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        # Load metadata
        with meta_path.open('r') as f:
            metadata = json.load(f)
        
        return {
            'rgb': rgb_image,
            'depth': depth_image,
            'mask': mask_image,
            'metadata': metadata,
            'frame_id': frame_id
        }
    
    def create_point_cloud(self, frame_data: dict, mask_filter: bool = True) -> o3d.geometry.PointCloud:
        """
        Generate point cloud from frame data.
        
        Args:
            frame_data: Frame data dictionary from load_frame_data
            mask_filter: If True, only include pixels where mask > 0
            
        Returns:
            Open3D point cloud
        """
        rgb = frame_data['rgb']
        depth = frame_data['depth']
        mask = frame_data['mask']
        metadata = frame_data['metadata']
        
        # Get camera intrinsics
        intrinsics = metadata['intrinsics']
        fx = intrinsics['fx']
        fy = intrinsics['fy']
        cx = intrinsics['ppx']
        cy = intrinsics['ppy']
        
        # Convert depth to meters
        depth_scale = metadata['depth_scale_m']
        depth_m = depth.astype(np.float32) * depth_scale
        
        # Create coordinate grids
        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply mask filter if requested
        if mask_filter:
            valid_mask = (mask > 0) & (depth_m > 0)
        else:
            valid_mask = depth_m > 0
        
        # Get valid pixels
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth_m[valid_mask]
        rgb_valid = rgb[valid_mask]
        
        # Convert to 3D coordinates
        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid
        
        # Create point cloud
        points = np.column_stack((x, y, z))
        colors = rgb_valid.astype(np.float32) / 255.0
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def process_all_frames(self, mask_filter: bool = True, save_individual: bool = True) -> List[o3d.geometry.PointCloud]:
        """
        Process all captured frames and generate point clouds.
        
        Args:
            mask_filter: Only include masked pixels (object regions)
            save_individual: Save each point cloud as separate .ply file
            
        Returns:
            List of generated point clouds
        """
        # Find all frame IDs
        frame_ids = []
        for rgb_file in self.rgb_dir.glob("*.png"):
            frame_id = rgb_file.stem
            frame_ids.append(frame_id)
        
        frame_ids.sort()
        print(f"Found {len(frame_ids)} frames to process")
        
        point_clouds = []
        successful = 0
        
        for frame_id in frame_ids:
            print(f"Processing frame {frame_id}...")
            
            # Load frame data
            frame_data = self.load_frame_data(frame_id)
            if frame_data is None:
                continue
            
            # Check if frame has detections
            detections = frame_data['metadata'].get('detections', [])
            if not detections:
                print(f"  Skipping {frame_id} - no detections")
                continue
            
            # Generate point cloud
            try:
                pcd = self.create_point_cloud(frame_data, mask_filter=mask_filter)
                
                if len(pcd.points) == 0:
                    print(f"  Warning: {frame_id} generated empty point cloud")
                    continue
                
                point_clouds.append(pcd)
                successful += 1
                
                print(f"  ✓ Generated point cloud with {len(pcd.points)} points")
                
                # Save individual point cloud
                if save_individual:
                    output_path = self.clouds_dir / f"{frame_id}.ply"
                    o3d.io.write_point_cloud(str(output_path), pcd)
                    print(f"  💾 Saved to {output_path.name}")
                    
            except Exception as e:
                print(f"  ✗ Error processing {frame_id}: {e}")
                continue
        
        print(f"\n🎉 Successfully processed {successful}/{len(frame_ids)} frames")
        print(f"Point clouds saved to: {self.clouds_dir}")
        
        return point_clouds
    
    def merge_point_clouds(self, point_clouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        """Merge multiple point clouds into one."""
        if not point_clouds:
            return o3d.geometry.PointCloud()
        
        print(f"Merging {len(point_clouds)} point clouds...")
        
        merged = o3d.geometry.PointCloud()
        for pcd in point_clouds:
            merged += pcd
        
        print(f"Merged point cloud has {len(merged.points)} points")
        return merged
    
    def clean_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Clean point cloud by removing outliers."""
        print("Cleaning point cloud...")
        
        # Remove statistical outliers
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        print(f"Removed {len(pcd.points) - len(pcd_clean.points)} outlier points")
        
        return pcd_clean
    
    def visualize_point_cloud(self, pcd: o3d.geometry.PointCloud, window_name: str = "Point Cloud"):
        """Visualize point cloud with Open3D."""
        print(f"Visualizing point cloud with {len(pcd.points)} points...")
        print("Controls: Mouse to rotate, scroll to zoom, ESC to exit")
        o3d.visualization.draw_geometries([pcd], window_name=window_name)


def main():
    """Main function to generate 3D point clouds from captured data."""
    print("=== 3D Point Cloud Generator ===")
    
    # Path to captured data
    captures_dir = Path(__file__).parent.parent / "captures"
    
    if not captures_dir.exists():
        print(f"❌ Captures directory not found: {captures_dir}")
        print("Run the capture script first to generate data")
        return
    
    # Create generator
    generator = PointCloudGenerator(captures_dir)
    
    print(f"Processing captures from: {captures_dir}")
    
    # Process all frames with mask filtering (object-only point clouds)
    print("\n1. Generating object-only point clouds (with mask filtering)...")
    object_clouds = generator.process_all_frames(mask_filter=True, save_individual=True)
    
    if object_clouds:
        # Merge all object point clouds
        print("\n2. Merging object point clouds...")
        merged_objects = generator.merge_point_clouds(object_clouds)
        
        # Clean merged point cloud
        cleaned = generator.clean_point_cloud(merged_objects)
        
        # Save merged point cloud
        merged_path = generator.clouds_dir / "merged_objects.ply"
        o3d.io.write_point_cloud(str(merged_path), cleaned)
        print(f"💾 Saved merged point cloud: {merged_path}")
        
        # Visualize result
        print("\n3. Visualizing merged object point cloud...")
        generator.visualize_point_cloud(cleaned, "Cup Object - 3D Point Cloud")
    else:
        print("❌ No point clouds generated. Check your captured data.")


if __name__ == "__main__":
    main()