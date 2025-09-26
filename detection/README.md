# Detection Package

A complete pipeline for RealSense-based object detection and 3D reconstruction.

## Package Structure

```
detection/
├── capture/           # RGB-D data capture
│   ├── realsense_yolo_848x480.py  # Main capture script with YOLO+SAM2
│   └── realsense.py               # Basic RealSense utilities
├── processing/        # Point cloud processing
│   ├── linemod_to_pointcloud.py     # Single frame processor
│   ├── batch_linemod_to_pointcloud.py  # Batch processor
│   ├── point_cloud.py             # Point cloud utilities
│   └── icp.py                     # ICP registration
├── reconstruction/    # 3D mesh reconstruction (coming soon)
├── utils/            # Common utilities
│   └── images_processing.py
├── tests/            # Test data and debugging
│   ├── create_test_data.py       # Generate synthetic data
│   └── debug_linemod.py          # Debug utilities
└── __main__.py       # Main CLI entry point
```

## Quick Start

### Using the main CLI

```bash
# Generate test data
python -m detection test-data --frames 5

# Capture RGB-D data
python -m detection capture --dataset-dir data/my_object --auto-capture

# Process single frame
python -m detection process data/my_object frame_001 --output-dir clouds/ --visualize

# Batch process all frames  
python -m detection batch data/my_object --output-dir clouds/ --voxel-size 0.002
```

### Using individual scripts

## RealSense + YOLO dataset capture

Use the capture module to preview D456 streams, capture paired RGB/Depth frames, and persist LINEMOD-style metadata:

```bash
python capture/realsense_yolo_848x480.py --dataset-dir data/my_capture --auto-capture
```

Press `c` to grab a frame manually, or add `--auto-capture` to save frames on a timer (configured via `--capture-interval`).

### Optional SAM2 mask refinement

If you want segmentation masks derived from SAM2 instead of simple bounding boxes:

1. Install the extras (PyTorch first, then SAM2).
   ```bash
   pip install torch torchvision torchaudio
   pip install 'git+https://github.com/facebookresearch/sam2.git'
   ```
2. Download the desired SAM2 YAML and checkpoint from the official release (for example `sam2_hiera_t.yaml` + `sam2_hiera_t.pth`).
3. Launch the capture script with the new flags:
   ```bash
   python capture/realsense_yolo_848x480.py \
       --dataset-dir data/my_capture \
       --sam2-config /path/to/sam2_hiera_t.yaml \
       --sam2-checkpoint /path/to/sam2_hiera_t.pth \
       --sam2-device cuda
   ```

When the model loads successfully, masks saved under `mask/` will reflect SAM2 output. If SAM2 is unavailable or a prediction fails, the script falls back to the bounding-box fill so captures can continue uninterrupted.

## LINEMOD to Point Cloud conversion

Convert your captured LINEMOD-style dataset into filtered point clouds using RealSense intrinsics and depth scaling:

### Single frame processing

```bash
python processing/linemod_to_pointcloud.py data/my_capture 20250926_143022_123456 --output-dir pointclouds/ --visualize
```

### Batch processing all frames

```bash
python processing/batch_linemod_to_pointcloud.py data/my_capture --output-dir pointclouds/ --voxel-size 0.002
```

The scripts automatically:
- Load RGB, depth, mask, and metadata from LINEMOD structure
- Extract RealSense intrinsics from metadata
- Apply depth scaling (mm to meters) 
- Generate masked point clouds
- Filter with voxel downsampling and outlier removal
- Save as .pcd files for further processing

### Filtering options

- `--voxel-size 0.001`: Downsample resolution (meters)
- `--nb-neighbors 20`: Points to consider for outlier detection
- `--std-ratio 2.0`: Standard deviation threshold for outliers
- `--skip-existing`: Skip frames that already have .pcd files
- `--max-frames N`: Limit processing to first N frames