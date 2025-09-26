#!/usr/bin/env python3
"""
Batch process all frames in a LINEMOD dataset to point clouds.
"""

import argparse
from pathlib import Path
from typing import List
import re

from .linemod_to_pointcloud import process_single_frame


def find_frame_ids(dataset_dir: Path) -> List[str]:
    """Find all frame IDs by scanning the rgb directory."""
    rgb_dir = dataset_dir / "rgb"
    if not rgb_dir.exists():
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
    
    frame_ids = []
    for rgb_file in rgb_dir.glob("*.png"):
        # Extract frame ID (filename without extension)
        frame_id = rgb_file.stem
        frame_ids.append(frame_id)
    
    # Sort frame IDs naturally (handles timestamps correctly)
    frame_ids.sort()
    return frame_ids


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert LINEMOD dataset frames to point clouds"
    )
    parser.add_argument(
        "dataset_dir", type=Path, help="Directory containing rgb/, depth/, mask/, meta/"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory to save point clouds"
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
        "--max-frames", type=int, help="Limit number of frames to process"
    )
    parser.add_argument(
        "--skip-existing", action="store_true", help="Skip frames that already have .pcd files"
    )
    
    args = parser.parse_args()
    
    # Find all frame IDs
    try:
        frame_ids = find_frame_ids(args.dataset_dir)
        print(f"Found {len(frame_ids)} frames in dataset")
    except Exception as e:
        print(f"Error scanning dataset: {e}")
        return 1
    
    # Limit frames if requested
    if args.max_frames:
        frame_ids = frame_ids[:args.max_frames]
        print(f"Processing first {len(frame_ids)} frames")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each frame
    success_count = 0
    skip_count = 0
    
    for i, frame_id in enumerate(frame_ids, 1):
        print(f"\n[{i}/{len(frame_ids)}] Processing {frame_id}")
        
        # Check if output already exists
        output_path = args.output_dir / f"{frame_id}.pcd"
        if args.skip_existing and output_path.exists():
            print(f"Skipping {frame_id} (already exists)")
            skip_count += 1
            continue
        
        try:
            pcd = process_single_frame(
                dataset_dir=args.dataset_dir,
                frame_id=frame_id,
                output_dir=args.output_dir,
                voxel_size=args.voxel_size,
                nb_neighbors=args.nb_neighbors,
                std_ratio=args.std_ratio,
                visualize=False
            )
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {frame_id}: {e}")
            continue
    
    print(f"\n=== Batch Processing Complete ===")
    print(f"Successful: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Failed: {len(frame_ids) - success_count - skip_count}")
    print(f"Output directory: {args.output_dir}")
    
    return 0


if __name__ == "__main__":
    exit(main())