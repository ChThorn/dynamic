import argparse
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


def capture_point_cloud(frame_settle_count: int = 5) -> Optional[o3d.geometry.PointCloud]:
    """Capture a single RGB-D frame from an Intel RealSense D456 and convert it to an Open3D point cloud."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

    align_to_color = rs.align(rs.stream.color)

    try:
        profile = pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        depth_units = depth_sensor.get_depth_scale()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()

        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width,
            intrinsics.height,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.ppx,
            intrinsics.ppy,
        )

        depth_scale = 1.0 / depth_units

        for _ in range(frame_settle_count):
            pipeline.wait_for_frames()

        depth_frame = None
        color_frame = None
        for _ in range(30):
            frames = pipeline.wait_for_frames()
            aligned_frames = align_to_color.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if depth_frame and color_frame:
                break

        if not depth_frame or not color_frame:
            return None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        color_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        depth_o3d = o3d.geometry.Image(depth_image)
        color_o3d = o3d.geometry.Image(color_rgb)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False,
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d_intrinsics)
        pcd.transform(
            np.array(
                [
                    [1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            )
        )

        return pcd
    finally:
        with contextlib.suppress(RuntimeError):
            pipeline.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture a single RealSense frame as an Open3D point cloud")
    parser.add_argument(
        "--frame-settle",
        type=int,
        default=5,
        help="Number of frames to discard before capture to let auto-exposure settle",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="File path or directory to save the captured point cloud (.pcd). If a directory is provided, a timestamped file is created",
    )
    parser.add_argument(
        "--label",
        default="capture",
        help="Filename prefix when --output points to a directory (default: capture)",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Skip on-screen visualization (useful for batch capture)",
    )
    return parser.parse_args()


def resolve_output_path(output: Optional[Path], label: str) -> Optional[Path]:
    if output is None:
        return None

    output = output.expanduser()

    if output.is_dir() or output.suffix == "":
        output.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return output / f"{label}_{timestamp}.pcd"

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != ".pcd":
        output = output.with_suffix(".pcd")
    return output


def main() -> None:
    args = parse_args()

    point_cloud = capture_point_cloud(frame_settle_count=args.frame_settle)
    if point_cloud is None or point_cloud.is_empty():
        print("Failed to capture a valid RGB-D frame; point cloud is empty.")
        return

    save_path = resolve_output_path(args.output, args.label)
    if save_path:
        o3d.io.write_point_cloud(str(save_path), point_cloud)
        print(f"Saved point cloud to {save_path}")

    if args.no_view:
        return

    o3d.visualization.draw_geometries([point_cloud], window_name="RealSense D456 Point Cloud")


if __name__ == "__main__":
    main()
