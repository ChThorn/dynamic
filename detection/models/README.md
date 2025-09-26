# Detection Models Directory

This directory contains pre-trained models for object detection and segmentation within the detection module.

## Directory Structure

```
detection/models/
├── yolo/              # YOLO detection models
│   ├── yolov8n.pt     # Nano model (fastest, least accurate)
│   ├── yolov8s.pt     # Small model (balanced)
│   ├── yolov8m.pt     # Medium model (more accurate)
│   ├── yolov8l.pt     # Large model (high accuracy)
│   └── yolov8x.pt     # Extra Large model (highest accuracy)
├── sam2/              # SAM2 segmentation models
│   ├── configs/       # SAM2 configuration files
│   │   ├── sam2_hiera_tiny.yaml
│   │   ├── sam2_hiera_small.yaml
│   │   ├── sam2_hiera_base_plus.yaml
│   │   └── sam2_hiera_large.yaml
│   └── checkpoints/   # SAM2 model weights
│       ├── sam2_hiera_tiny.pth
│       ├── sam2_hiera_small.pth
│       ├── sam2_hiera_base_plus.pth
│       └── sam2_hiera_large.pth
├── model_manager.py   # Model download utility
└── README.md          # This file
```

## Quick Setup

### Download Basic Models (Recommended)
```bash
# Navigate to project root
cd /path/to/dynamic

# Download basic models for development
python -m detection.models.model_manager --setup-basic
```

This downloads:
- `yolov8n.pt` (6MB) - Fastest YOLO model for real-time use
- `sam2_hiera_tiny.yaml` + `sam2_hiera_tiny.pth` (38MB) - Fastest SAM2 model

### Check Model Status
```bash
python -m detection.models.model_manager --status
```

## Model Details

### YOLO Models

YOLO models are automatically downloaded by ultralytics when first used, but you can pre-download them:

```bash
# Download specific YOLO models
python -m detection.models.model_manager --yolo yolov8n  # Fast
python -m detection.models.model_manager --yolo yolov8s  # Balanced  
python -m detection.models.model_manager --yolo yolov8m  # Accurate
```

| Model    | Size  | Speed | mAP@0.5:0.95 | Use Case |
|----------|-------|-------|--------------|----------|
| YOLOv8n  | 6MB   | Fast  | 37.3         | Real-time, embedded |
| YOLOv8s  | 22MB  | Good  | 44.9         | Balanced performance |
| YOLOv8m  | 52MB  | OK    | 50.2         | Higher accuracy |
| YOLOv8l  | 87MB  | Slow  | 52.9         | Production quality |
| YOLOv8x  | 136MB | Slower| 53.9         | Best accuracy |

### SAM2 Models

```bash
# Download specific SAM2 models (config + checkpoint)
python -m detection.models.model_manager --sam2 sam2_hiera_tiny      # Fast
python -m detection.models.model_manager --sam2 sam2_hiera_small     # Balanced
python -m detection.models.model_manager --sam2 sam2_hiera_base_plus # Accurate
python -m detection.models.model_manager --sam2 sam2_hiera_large     # Best
```

| Model          | Size  | Speed | Quality | Use Case |
|----------------|-------|-------|---------|----------|
| sam2_hiera_tiny| 38MB  | Fast  | Good    | Real-time applications |
| sam2_hiera_small| 184MB| OK    | Better  | Balanced performance |
| sam2_hiera_base_plus| 319MB| Slow | High   | Production quality |
| sam2_hiera_large| 894MB| Slower| Highest | Best segmentation |

## Usage in Capture Script

The capture script automatically uses models from this directory:

```python
# Default configuration (in realsense_yolo_848x480.py)
YOLO_MODEL_PATH = detection/models/yolo/yolov8n.pt
SAM2_CONFIG = detection/models/sam2/configs/sam2_hiera_tiny.yaml  
SAM2_CHECKPOINT = detection/models/sam2/checkpoints/sam2_hiera_tiny.pth
```

## Model Selection Guidelines

### For Real-time Applications (RealSense capture)
- **YOLO**: yolov8n.pt or yolov8s.pt
- **SAM2**: sam2_hiera_tiny.pth or sam2_hiera_small.pth

### For High Accuracy Applications
- **YOLO**: yolov8l.pt or yolov8x.pt  
- **SAM2**: sam2_hiera_base_plus.pth or sam2_hiera_large.pth

### For Development/Testing
- **YOLO**: yolov8n.pt (fastest download and inference)
- **SAM2**: sam2_hiera_tiny.pth (smallest and fastest)

## Git Considerations

Large model files are excluded from git in `.gitignore`:
```
# Large model files (download via model_manager.py)
detection/models/yolo/*.pt
detection/models/sam2/checkpoints/*.pth
```

Config files and this README are committed since they're small and useful for setup.

## Troubleshooting

### Models Not Found
```bash
# Check what's downloaded
python -m detection.models.model_manager --status

# Download missing models
python -m detection.models.model_manager --setup-basic
```

### Internet Connection Issues
- Models are downloaded from official sources (GitHub, Facebook AI)
- Large files may take time depending on connection speed
- Failed downloads are automatically cleaned up

### CUDA/Device Issues
- YOLO works on CPU and GPU automatically
- SAM2 device can be configured: `cuda`, `cuda:0`, `cpu`
- If CUDA unavailable, set `SAM2_DEVICE = "cpu"` in capture script