# Examples Directory Guide

This directory contains the essential examples and demos for the robot motion planning system.

## 🎯 Main Files (What You Need):

### **Core Integration Demo:**
- **`test_poses_selection.py`** - Interactive pose selection with visualizer → robot execution
  - Use this for testing complete workflow with real robot
  - Select poses with visualizer, execute motions

### **Essential Examples:**
- **`examples.py`** - Core motion planning examples and validation
- **`safety_first_example.py`** - Safety system demonstration
- **`validation_summary.py`** - Complete system validation report

### **Performance Analysis:**
- **`test_polynomial_trajectory.py`** - Polynomial vs linear trajectory comparison

### **Integration Bridge:**
- **`integrated_pose_example.py`** - Complete integration workflow example

### **Documentation:**
- **`architecture_summary.py`** - System architecture overview

## 🚀 Quick Start:

For **complete workflow with real robot**:
```bash
python3 test_poses_selection.py
```

For **validation and examples**:
```bash
python3 validation_summary.py
```

For **safety demonstration**:
```bash
python3 safety_first_example.py
```

## 📊 Generated Files:
- `pick_place_poses.json` - Poses selected from visualizer
- `*.png` - Performance comparison plots

---
**Note**: All redundant test files have been removed to keep the directory clean and focused.