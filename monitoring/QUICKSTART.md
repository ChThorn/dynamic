# Monitoring Package - Quick Start

## Package Structure
```
monitoring/
├── __init__.py                 # Package initialization and imports
├── AdvancedPoseVisualizer.py   # Interactive 3D pose control tool
├── calibration_data/           # Self-contained calibration data
│   ├── improved_calibration_results.json  # Eye-to-hand calibration
│   └── README.md              # Calibration documentation
├── demo.py                     # Example usage and demo script
├── README.md                   # Detailed documentation
├── QUICKSTART.md              # This quick start guide
└── *.json                     # Exported pose data files
```

## Key Features

✅ **Self-Contained**: No external dependencies on other packages  
✅ **Local Calibration**: Eye-to-hand calibration data included  
✅ **Production Ready**: Robust error handling and logging  
✅ **Easy Integration**: Compatible with motion planning workflows  

## Quick Usage

### 1. Direct Usage
```bash
cd monitoring/
python3 AdvancedPoseVisualizer.py
```

### 2. Package Import
```python
from monitoring import AdvancedPoseVisualizer

visualizer = AdvancedPoseVisualizer()
visualizer.run()
poses = visualizer.get_poses()
```

### 3. Demo Script
```bash
cd monitoring/
python3 demo.py
```

## Integration with Motion Planning

The monitoring package is designed to work seamlessly with the motion planning system:

```python
# 1. Define poses interactively
from monitoring import AdvancedPoseVisualizer
visualizer = AdvancedPoseVisualizer()
visualizer.run()
target_poses = visualizer.get_poses()

# 2. Plan motions (future integration)
from planning.src.motion_planner import MotionPlanner
planner = MotionPlanner()

for i, pose in enumerate(target_poses[:-1]):
    result = planner.plan_cartesian_motion(pose, target_poses[i+1])
    print(f"Motion {i+1}: {result.status}")
```

## Features Demonstrated

✅ **Self-Contained Package**: No dependencies on charuco or other packages  
✅ **Local Calibration Data**: Eye-to-hand calibration included locally  
✅ **Interactive Pose Control**: 3D visualization with sliders  
✅ **JSON Export**: Compatible with motion planning  
✅ **Error Handling**: Graceful fallbacks for missing files  
✅ **Production Ready**: Proper logging and documentation  

## Next Steps

The monitoring package is ready for:
1. **Standalone Operation**: Complete independence from charuco package
2. **Integration with Motion Planning**: Seamless workflow integration
3. **Extension with Additional Tools**: Trajectory monitoring, performance analysis
4. **Production Deployment**: Industrial robotics applications

Package successfully created and tested! 🎉