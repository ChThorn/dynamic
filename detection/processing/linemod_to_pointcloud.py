#!/usr/bin/env python3
"""
Convert LINEMOD-style captures (RGB + depth + mask + metadata) into filtered point clouds.
Handles RealSense intrinsics and depth scaling automatically.
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import open3d as o3d


def load_linemod_frame(
    dataset_dir: Path, frame_id: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load RGB, depth, mask, and metadata for a single frame."""
    rgb_path = dataset_dir / "rgb" / f"{frame_id}.png"
    depth_path = dataset_dir / "depth" / f"{frame_id}.png"
    mask_path = dataset_dir / "mask" / f"{frame_id}.png"
    meta_path = dataset_dir / "meta" / f"{frame_id}.json"
    
    # Load images
    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth image not found: {depth_path}")
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask image not found: {mask_path}")
    
    # Load metadata
    with meta_path.open('r') as f:
        metadata = json.load(f)
    
    return rgb, depth, mask, metadata


def create_intrinsic_matrix(metadata: dict) -> o3d.camera.PinholeCameraIntrinsic:
    """Create Open3D intrinsic matrix from metadata."""
    intrinsics_data = metadata["intrinsics"]
    return o3d.camera.PinholeCameraIntrinsic(
        width=intrinsics_data["width"],
        height=intrinsics_data["height"],
        fx=intrinsics_data["fx"],
        fy=intrinsics_data["fy"],
        cx=intrinsics_data["ppx"],
        cy=intrinsics_data["ppy"],
    )


def apply_mask_to_rgbd(
    rgb: np.ndarray, 
    depth: np.ndarray, 
    mask: np.ndarray, 
    depth_scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply mask to RGB and depth, convert depth to meters."""
    # Convert depth from mm to meters
    depth_m = depth.astype(np.float32) * depth_scale
    
    # Apply mask (set non-masked pixels to 0)
    mask_bool = mask > 127  # threshold for binary mask
    
    masked_rgb = rgb.copy()
    masked_depth = depth_m.copy()
    
    # Zero out pixels outside the mask
    masked_rgb[~mask_bool] = 0
    masked_depth[~mask_bool] = 0.0
    
    return masked_rgb, masked_depth


def rgbd_to_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    depth_scale: Optional[float] = None
) -> o3d.geometry.PointCloud:
    """Convert masked RGB-D to point cloud."""
    # Create Open3D images
    o3d_rgb = o3d.geometry.Image(rgb.astype(np.uint8))
    
    # Depth is already in meters, convert to uint16 millimeters for Open3D
    depth_mm = (depth * 1000).astype(np.uint16)
    o3d_depth = o3d.geometry.Image(depth_mm)
    
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=1000.0, convert_rgb_to_intensity=False
    )
    
    # Generate point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    
    return pcd


def filter_pointcloud(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = 0.001,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    remove_outliers: bool = True
) -> o3d.geometry.PointCloud:
    """Apply filtering to clean up the point cloud."""
    # Remove points with zero coordinates (from masked regions)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    # Filter out zero points
    valid_mask = np.any(points != 0, axis=1)
    if np.any(valid_mask):
        pcd.points = o3d.utility.Vector3dVector(points[valid_mask])
        pcd.colors = o3d.utility.Vector3dVector(colors[valid_mask])
    
    if len(pcd.points) == 0:
        print("Warning: Point cloud is empty after filtering zero points")
        return pcd
    
    # Voxel downsampling
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"After voxel downsampling: {len(pcd.points)} points")
    
    # Statistical outlier removal
    if remove_outliers and len(pcd.points) > nb_neighbors:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        print(f"After outlier removal: {len(pcd.points)} points")
    
    return pcd


def process_single_frame(
    dataset_dir: Path,
    frame_id: str,
    output_dir: Optional[Path] = None,
    voxel_size: float = 0.001,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    visualize: bool = False
) -> o3d.geometry.PointCloud:
    """Process a single LINEMOD frame into a filtered point cloud."""
    print(f"Processing frame {frame_id}...")
    
    # Load data
    rgb, depth, mask, metadata = load_linemod_frame(dataset_dir, frame_id)
    
    # Extract depth scale (stored in metadata)
    depth_scale_m = metadata.get("depth_scale_m", 0.001)  # Default 1mm = 0.001m
    
    # Create intrinsic matrix
    intrinsic = create_intrinsic_matrix(metadata)
    
    # Apply mask and convert depth units
    masked_rgb, masked_depth = apply_mask_to_rgbd(rgb, depth, mask, depth_scale_m)
    
    # Convert to point cloud
    pcd = rgbd_to_pointcloud(masked_rgb, masked_depth, intrinsic)
    
    # Filter point cloud
    pcd = filter_pointcloud(pcd, voxel_size, nb_neighbors, std_ratio)
    
    print(f"Final point cloud: {len(pcd.points)} points")
    
    # Save if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{frame_id}.pcd"
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"Saved to {output_path}")
    
    # Visualize if requested
    if visualize:
        o3d.visualization.draw_geometries([pcd], window_name=f"Frame {frame_id}")
    
    return pcd


def main():
    parser = argparse.ArgumentParser(
        description="Convert LINEMOD-style captures to filtered point clouds"
    )
    parser.add_argument(
        "dataset_dir", type=Path, help="Directory containing rgb/, depth/, mask/, meta/"
    )
    parser.add_argument(
        "frame_id", help="Frame ID to process (e.g., '20250926_143022_123456')"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Directory to save point cloud (.pcd files)"
    )
    parser.add_argument(
        "--voxel-size", type=float, default=0.001, help="Voxel size for downsampling (m)"
    )
    parser.add_argument(
        "--nb-neighbors", type=int, default=20, help="Number of neighbors for outlier removal"
    )
    parser.add_argument(
        "--std-ratio", type=float, default=2.0, help="Standard deviation ratio for outlier removal"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Show point cloud in 3D viewer"
    )
    
    args = parser.parse_args()
    
    try:
        pcd = process_single_frame(
            dataset_dir=args.dataset_dir,
            frame_id=args.frame_id,
            output_dir=args.output_dir,
            voxel_size=args.voxel_size,
            nb_neighbors=args.nb_neighbors,
            std_ratio=args.std_ratio,
            visualize=args.visualize
        )
        
        print(f"Successfully processed frame {args.frame_id}")
        
    except Exception as e:
        print(f"Error processing frame {args.frame_id}: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())