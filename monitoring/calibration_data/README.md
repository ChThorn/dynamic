# Calibration Data

This directory contains eye-to-hand calibration data for the robot system.

## Files

### improved_calibration_results.json
Eye-to-hand calibration results from ChArUco board calibration performed on August 22, 2025.

**Transformations:**
- `T_C_R`: Camera to Robot transformation matrix (4x4 homogeneous)
- `T_T_B`: Tool to Base transformation matrix (4x4 homogeneous)

**Validation Metrics:**
- Translation accuracy and rotation accuracy metrics
- Validation performed with real robot hardware

**Original Source:**
Copied from charuco/calibration_20250822_161105/ for self-contained monitoring package.

## Usage in AdvancedPoseVisualizer

The calibration data is automatically loaded when the AdvancedPoseVisualizer is initialized:

```python
# Automatic loading
visualizer = AdvancedPoseVisualizer()  # Uses default calibration

# Explicit path
visualizer = AdvancedPoseVisualizer("custom_calibration.json")
```

## Coordinate Systems

- **Robot Base Frame**: Origin at robot base
- **Camera Frame**: From eye-to-hand calibration setup
- **TCP Frame**: Tool Center Point frame

The transformation matrices enable conversion between these coordinate systems for accurate pose visualization and robot control.