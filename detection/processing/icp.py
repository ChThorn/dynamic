import argparse
import copy
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))
    if pcd.is_empty():
        raise ValueError(f"Point cloud '{path}' is empty or could not be loaded.")
    return pcd


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float) -> o3d.geometry.PointCloud:
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2 if voxel_size > 0 else 0.05
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(50)
    return pcd


def run_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    threshold: float,
    initial_transform: np.ndarray,
    point_to_plane: bool,
) -> o3d.pipelines.registration.RegistrationResult:
    estimation = (
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
        if point_to_plane
        else o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        initial_transform,
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )


def visualize_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
) -> None:
    source_transformed = copy.deepcopy(source)
    source_transformed.transform(transformation)
    source_transformed.paint_uniform_color([1, 0.706, 0])  # orange
    target_temp = copy.deepcopy(target)
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # blue
    o3d.visualization.draw_geometries([source_transformed, target_temp], window_name="ICP Alignment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust ICP alignment using Open3D")
    parser.add_argument("source", type=Path, help="Path to source point cloud (.pcd/.ply)")
    parser.add_argument("target", type=Path, help="Path to target point cloud (.pcd/.ply)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Maximum correspondence distance (meters)")
    parser.add_argument("--voxel", type=float, default=0.01, help="Voxel size for downsampling (meters)")
    parser.add_argument(
        "--point-to-plane",
        action="store_true",
        help="Use point-to-plane estimation (requires normals; more accurate for structured surfaces)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip alignment visualization",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"Source point cloud '{args.source}' not found.")
    if not args.target.exists():
        raise FileNotFoundError(f"Target point cloud '{args.target}' not found.")

    source = load_point_cloud(args.source)
    target = load_point_cloud(args.target)

    source_pre = preprocess_point_cloud(source, args.voxel)
    target_pre = preprocess_point_cloud(target, args.voxel)

    initial_transform = np.eye(4)

    result = run_icp(
        source_pre,
        target_pre,
        threshold=args.threshold,
        initial_transform=initial_transform,
        point_to_plane=args.point_to_plane,
    )

    fitness = result.fitness
    rmse = result.inlier_rmse

    print("ICP finished with:")
    print(f"  Fitness: {fitness:.4f}")
    print(f"  RMSE:    {rmse:.4f}")
    print("Transformation matrix:")
    print(result.transformation)

    if not args.no_visualize:
        visualize_registration(source, target, result.transformation)


if __name__ == "__main__":
    main()
