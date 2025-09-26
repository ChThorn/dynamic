#!/usr/bin/env python3
"""
Enhanced 3D Point Cloud Generator with better quality settings.

This version addresses common quality issues by:
- Offering both precise and expanded mask modes
- Better depth filtering
- Enhanced point cloud processing
- Multiple visualization options
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d


class EnhancedPointCloudGenerator:
    """Enhanced point cloud generator with quality improvements."""
    
    def __init__(self, captures_dir: Path):
        self.captures_dir = Path(captures_dir)
        self.rgb_dir = self.captures_dir / "rgb"
        self.depth_dir = self.captures_dir / "depth"
        self.mask_dir = self.captures_dir / "mask"
        self.meta_dir = self.captures_dir / "meta"
        
        self.clouds_dir = self.captures_dir.parent / "clouds"
        self.clouds_dir.mkdir(exist_ok=True)
    
    def load_frame_data(self, frame_id: str) -> Optional[dict]:
        """Load all data for a specific frame."""
        rgb_path = self.rgb_dir / f"{frame_id}.png"
        depth_path = self.depth_dir / f"{frame_id}.png"
        mask_path = self.mask_dir / f"{frame_id}.png"
        meta_path = self.meta_dir / f"{frame_id}.json"
        
        if not all(p.exists() for p in [rgb_path, depth_path, mask_path, meta_path]):
            return None
        
        rgb_image = cv2.imread(str(rgb_path))
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        depth_image = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
        mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        with meta_path.open('r') as f:
            metadata = json.load(f)
        
        return {
            'rgb': rgb_image,
            'depth': depth_image,
            'mask': mask_image,
            'metadata': metadata,
            'frame_id': frame_id
        }
    
    def expand_mask(self, mask: np.ndarray, expansion_pixels: int = 5) -> np.ndarray:
        """Expand mask to include more surrounding pixels."""
        if expansion_pixels <= 0:
            return mask
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                         (expansion_pixels*2+1, expansion_pixels*2+1))
        expanded = cv2.dilate(mask, kernel, iterations=1)
        return expanded
    
    def create_enhanced_point_cloud(self, frame_data: dict, 
                                  mask_mode: str = "expanded",
                                  depth_filter: bool = True,
                                  expansion_pixels: int = 8) -> o3d.geometry.PointCloud:
        """
        Generate enhanced point cloud with better quality.
        
        Args:
            frame_data: Frame data dictionary
            mask_mode: "precise", "expanded", or "bounding_box" 
            depth_filter: Apply depth-based filtering
            expansion_pixels: Pixels to expand mask (if expanded mode)
        """
        rgb = frame_data['rgb']
        depth = frame_data['depth']
        mask = frame_data['mask']
        metadata = frame_data['metadata']
        
        # Get camera intrinsics
        intrinsics = metadata['intrinsics']
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['ppx'], intrinsics['ppy']
        
        # Convert depth to meters
        depth_scale = metadata['depth_scale_m']
        depth_m = depth.astype(np.float32) * depth_scale
        
        # Create mask based on mode
        if mask_mode == "precise":
            final_mask = mask > 0
        elif mask_mode == "expanded":
            expanded_mask = self.expand_mask(mask, expansion_pixels)
            final_mask = expanded_mask > 0
        elif mask_mode == "generous":
            generous_mask = self.expand_mask(mask, expansion_pixels)
            final_mask = generous_mask > 0
        elif mask_mode == "bounding_box":
            # Create bounding box mask from detections
            final_mask = np.zeros_like(mask, dtype=bool)
            for det in metadata.get('detections', []):
                x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                final_mask[y1:y2, x1:x2] = True
        
        # Apply depth filtering
        if depth_filter:
            # Remove pixels that are too close/far
            depth_valid = (depth_m > 0.1) & (depth_m < 3.0)  # 10cm to 3m range
            final_mask = final_mask & depth_valid
        else:
            final_mask = final_mask & (depth_m > 0)
        
        # Create coordinate grids
        height, width = depth.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Get valid pixels
        u_valid = u[final_mask]
        v_valid = v[final_mask]
        depth_valid = depth_m[final_mask]
        rgb_valid = rgb[final_mask]
        
        # Convert to 3D coordinates
        x = (u_valid - cx) * depth_valid / fx
        y = (v_valid - cy) * depth_valid / fy
        z = depth_valid
        
        # Create point cloud
        points = np.column_stack((x, y, z))
        colors = rgb_valid.astype(np.float32) / 255.0
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd
    
    def process_with_quality_modes(self) -> dict:
        """Process all frames with different quality modes."""
        # Find frame IDs
        frame_ids = []
        for rgb_file in self.rgb_dir.glob("*.png"):
            frame_id = rgb_file.stem
            frame_ids.append(frame_id)
        frame_ids.sort()
        
        results = {}
        modes = {
            "precise": {"expansion_pixels": 0, "desc": "SAM2 precise masks"},
            "expanded": {"expansion_pixels": 8, "desc": "Expanded masks (+8px)"},
            "generous": {"expansion_pixels": 15, "desc": "Generous masks (+15px)"},
            "bounding_box": {"expansion_pixels": 0, "desc": "Full bounding boxes"}
        }
        
        for mode_name, settings in modes.items():
            print(f"\n=== Processing with {settings['desc']} ===")
            
            point_clouds = []
            successful = 0
            
            for frame_id in frame_ids:
                frame_data = self.load_frame_data(frame_id)
                if frame_data is None or not frame_data['metadata'].get('detections'):
                    continue
                
                try:
                    pcd = self.create_enhanced_point_cloud(
                        frame_data, 
                        mask_mode=mode_name,
                        expansion_pixels=settings['expansion_pixels']
                    )
                    
                    if len(pcd.points) > 100:  # Minimum point threshold
                        point_clouds.append(pcd)
                        successful += 1
                        
                except Exception as e:
                    print(f"Error processing {frame_id}: {e}")
                    continue
            
            if point_clouds:
                # Merge and clean
                merged = self.merge_point_clouds(point_clouds)
                cleaned = self.clean_point_cloud(merged)
                
                # Save result
                output_path = self.clouds_dir / f"enhanced_{mode_name}.ply"
                o3d.io.write_point_cloud(str(output_path), cleaned)
                
                results[mode_name] = {
                    'point_cloud': cleaned,
                    'point_count': len(cleaned.points),
                    'frames_processed': successful,
                    'file_path': output_path
                }
                
                print(f"✓ {mode_name}: {len(cleaned.points)} points from {successful} frames")
                print(f"  Saved: {output_path.name}")
            else:
                results[mode_name] = None
                print(f"✗ {mode_name}: No valid point clouds generated")
        
        return results
    
    def merge_point_clouds(self, point_clouds: List[o3d.geometry.PointCloud]) -> o3d.geometry.PointCloud:
        """Merge multiple point clouds."""
        if not point_clouds:
            return o3d.geometry.PointCloud()
        
        merged = o3d.geometry.PointCloud()
        for pcd in point_clouds:
            merged += pcd
        return merged
    
    def clean_point_cloud(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """Enhanced point cloud cleaning."""
        if len(pcd.points) < 100:
            return pcd
        
        # Remove statistical outliers
        pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Remove radius outliers for denser cleaning  
        pcd_clean, _ = pcd_clean.remove_radius_outlier(nb_points=10, radius=0.02)
        
        return pcd_clean
    
    def compare_results(self, results: dict):
        """Compare results from different modes."""
        print(f"\n=== Quality Comparison ===")
        
        for mode_name, result in results.items():
            if result:
                pcd = result['point_cloud']
                bbox = pcd.get_axis_aligned_bounding_box()
                extent = bbox.get_extent()
                
                print(f"{mode_name:12}: {result['point_count']:6} points, "
                      f"size: {extent[0]:.3f}×{extent[1]:.3f}×{extent[2]:.3f}m")
        
        # Recommend best mode
        if results.get('expanded'):
            print(f"\n💡 Recommendation: 'expanded' mode usually provides the best balance")
            return results['expanded']['point_cloud']
        elif results.get('generous'):
            print(f"\n💡 Recommendation: 'generous' mode for maximum coverage")
            return results['generous']['point_cloud']
        else:
            print(f"\n💡 Using available result")
            for result in results.values():
                if result:
                    return result['point_cloud']
        
        return None


def main():
    print("=== Enhanced 3D Point Cloud Generator ===")
    
    captures_dir = Path(__file__).parent.parent / "captures"
    if not captures_dir.exists():
        print(f"❌ Captures directory not found: {captures_dir}")
        return
    
    generator = EnhancedPointCloudGenerator(captures_dir)
    
    # Process with all quality modes
    results = generator.process_with_quality_modes()
    
    # Compare and recommend
    best_pcd = generator.compare_results(results)
    
    if best_pcd:
        print(f"\n=== Visualizing Best Result ===")
        print("Controls: Mouse to rotate, scroll to zoom, ESC to exit")
        o3d.visualization.draw_geometries([best_pcd], window_name="Enhanced Cup Reconstruction")


if __name__ == "__main__":
    main()