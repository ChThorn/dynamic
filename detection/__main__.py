#!/usr/bin/env python3
"""
Simple CLI for the detection package.

Usage:
    python -m detection [options]

Options:
    --help, -h              Show this help message
    --capture DIR           Capture RGB-D data to directory
    --auto-capture          Enable automatic capture (with --capture)
    --sam2-config FILE      SAM2 config file path
    --sam2-checkpoint FILE  SAM2 checkpoint file path
    --process DIR FRAME     Process single frame to point cloud
    --batch DIR             Batch process all frames in directory  
    --output-dir DIR        Output directory for processing
    --voxel-size SIZE       Voxel size for downsampling (default: 0.001)
    --visualize             Show 3D visualization (with --process)
    --test-data [N]         Generate N test frames (default: 3)
    --skip-existing         Skip existing files in batch mode
"""

import argparse
import sys
from pathlib import Path


def print_help():
    """Print usage information."""
    print(__doc__.strip())


def main():
    if len(sys.argv) == 1 or '--help' in sys.argv or '-h' in sys.argv:
        print_help()
        return 0
    
    # Simple argument parsing without subparsers
    args = sys.argv[1:]
    
    try:
        # Capture mode
        if '--capture' in args:
            idx = args.index('--capture')
            if idx + 1 >= len(args):
                print("Error: --capture requires a directory")
                return 1
            
            dataset_dir = Path(args[idx + 1])
            auto_capture = '--auto-capture' in args
            
            # Optional SAM2 parameters
            sam2_config = None
            sam2_checkpoint = None
            if '--sam2-config' in args:
                sam2_idx = args.index('--sam2-config')
                if sam2_idx + 1 < len(args):
                    sam2_config = Path(args[sam2_idx + 1])
            
            if '--sam2-checkpoint' in args:
                sam2_idx = args.index('--sam2-checkpoint')
                if sam2_idx + 1 < len(args):
                    sam2_checkpoint = Path(args[sam2_idx + 1])
            
            # Call capture function
            from .capture.realsense_yolo_848x480 import main as capture_main
            capture_argv = [
                "realsense_yolo_848x480.py",
                f"--dataset-dir={dataset_dir}"
            ]
            
            if auto_capture:
                capture_argv.append("--auto-capture")
            if sam2_config:
                capture_argv.append(f"--sam2-config={sam2_config}")
            if sam2_checkpoint:
                capture_argv.append(f"--sam2-checkpoint={sam2_checkpoint}")
            
            old_argv = sys.argv[:]
            sys.argv = capture_argv
            result = capture_main()
            sys.argv = old_argv
            return result
        
        # Process single frame
        elif '--process' in args:
            idx = args.index('--process')
            if idx + 2 >= len(args):
                print("Error: --process requires dataset_dir and frame_id")
                return 1
            
            dataset_dir = Path(args[idx + 1])
            frame_id = args[idx + 2]
            
            # Optional parameters
            output_dir = None
            if '--output-dir' in args:
                out_idx = args.index('--output-dir')
                if out_idx + 1 < len(args):
                    output_dir = Path(args[out_idx + 1])
            
            voxel_size = 0.001
            if '--voxel-size' in args:
                vox_idx = args.index('--voxel-size')
                if vox_idx + 1 < len(args):
                    voxel_size = float(args[vox_idx + 1])
            
            visualize = '--visualize' in args
            
            from .processing.linemod_to_pointcloud import process_single_frame
            process_single_frame(
                dataset_dir=dataset_dir,
                frame_id=frame_id,
                output_dir=output_dir,
                voxel_size=voxel_size,
                visualize=visualize
            )
        
        # Batch process
        elif '--batch' in args:
            idx = args.index('--batch')
            if idx + 1 >= len(args):
                print("Error: --batch requires a directory")
                return 1
            
            dataset_dir = Path(args[idx + 1])
            
            # Output directory is required for batch
            if '--output-dir' not in args:
                print("Error: --batch requires --output-dir")
                return 1
            
            out_idx = args.index('--output-dir')
            if out_idx + 1 >= len(args):
                print("Error: --output-dir requires a directory")
                return 1
            
            output_dir = Path(args[out_idx + 1])
            
            # Optional parameters
            voxel_size = 0.001
            if '--voxel-size' in args:
                vox_idx = args.index('--voxel-size')
                if vox_idx + 1 < len(args):
                    voxel_size = float(args[vox_idx + 1])
            
            skip_existing = '--skip-existing' in args
            
            from .processing.batch_linemod_to_pointcloud import main as batch_main
            
            batch_argv = [
                "batch_linemod_to_pointcloud.py",
                str(dataset_dir),
                f"--output-dir={output_dir}",
                f"--voxel-size={voxel_size}"
            ]
            
            if skip_existing:
                batch_argv.append("--skip-existing")
            
            old_argv = sys.argv[:]
            sys.argv = batch_argv
            result = batch_main()
            sys.argv = old_argv
            return result
        
        # Generate test data
        elif '--test-data' in args:
            idx = args.index('--test-data')
            
            # Optional number of frames
            num_frames = 3
            if idx + 1 < len(args) and not args[idx + 1].startswith('--'):
                num_frames = int(args[idx + 1])
            
            # Default to detection/test_data/
            detection_dir = Path(__file__).parent
            output_dir = detection_dir / "test_data"
            
            from .tests.create_test_data import create_test_dataset
            create_test_dataset(output_dir, num_frames)
            print(f"Created {num_frames} test frames in {output_dir}")
        
        else:
            print("Error: No valid command specified")
            print("Use --help to see available options")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())