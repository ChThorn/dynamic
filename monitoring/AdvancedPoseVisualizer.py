#!/usr/bin/env python3
"""
Advanced 3D Interactive Pose Control Visualizer

This tool combines a multi-view 2D plotting interface for precise point definition
with the back-end logic for generating robot TCP poses, now with rotational adjustment.

Features:
- Self-contained with local calibration data (no external dependencies)
- Multi-view 2D plotting for position definition  
- Interactive sliders for orientation control
- Real-time 3D visualization with coordinate frames
- Export poses to JSON format
- Integration with robot calibration data

New Interaction Workflow:
1.  Click on the X-Y and Y-Z plots to define a point's position.
2.  A red "preview" point and a colored TCP frame appear.
3.  Drag the 2D markers or use 'e' for text boxes to adjust the POSITION.
4.  Use the new Roll, Pitch, and Yaw sliders to adjust the ORIENTATION.
5.  Press 'Enter' on the plot window to finalize the pose.

Author: Robot Motion Planning Team
Package: monitoring
Date: September 16, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Slider
from mpl_toolkits.mplot3d import Axes3D
import json
import logging
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import time
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("PoseVisualizer")


class AdvancedPoseVisualizer:
    """
    Combines the interactive 2D->3D plotter with robot pose generation logic,
    including interactive sliders for rotational adjustment.
    
    This tool is part of the monitoring package for the robot motion planning system.
    It provides an interactive interface for defining TCP poses that can be used
    with the motion planning system.
    """
    def __init__(self, calibration_path: str = None):
        # Default to local calibration data if no path provided
        if calibration_path is None:
            # Use local calibration data in monitoring package
            local_calibration = Path(__file__).parent / "calibration_data" / "improved_calibration_results.json"
            if local_calibration.exists():
                calibration_path = str(local_calibration)
            else:
                # Fallback to relative path (for backward compatibility)
                calibration_path = "calibration_data/improved_calibration_results.json"
        
        self.calibration_path = Path(calibration_path)
        self.load_calibration_data()
        self.load_workspace_constraints()
        
        # Parameters
        self.coordinate_frame_size = 0.1
        self.robot_poses = []
        self.orientation_mode = "downward"  # Default to downward for pick and place operations

        # Setup plot
        self.fig, self.axs = plt.subplot_mosaic(
            [["main", "main"], ["xy", "yz"]],
            figsize=(12, 10), # Increased height for sliders
            per_subplot_kw={"main": {'projection': '3d'}}
        )
        self.fig.subplots_adjust(bottom=0.25, top=0.95) # Make more space at the bottom
        self.ax_3d = self.axs["main"]
        self.ax_xy = self.axs["xy"]
        self.ax_yz = self.axs["yz"]
        
        # State management
        self.coords = {'x': None, 'y': None, 'z': None}
        self.orientation_rpy = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0} # In radians
        self.dragged_artist = None
        self.preview_point_visible = False
        
        # Setup UI
        self._configure_plots()
        self._create_markers()
        self._create_text_boxes()
        self._create_rotation_sliders() # NEW
        self._connect_events()
        self._draw_static_elements()
        self._create_legend()
        
        logger.info("Advanced Pick & Place Pose Visualizer with workspace constraints initialized.")

    def load_calibration_data(self):
        """Load calibration data with graceful fallback."""
        try:
            with open(self.calibration_path, 'r') as f:
                calibration_data = json.load(f)
            logger.info(f"Calibration data loaded from: {self.calibration_path}")
        except FileNotFoundError:
            logger.warning(f"Calibration file not found: {self.calibration_path}")
            logger.info("Using default transformation matrices for basic functionality")
            # Use identity matrices as fallback
            self.T_C_R = np.eye(4)
            self.T_T_B = np.eye(4)
            self.T_R_C = np.eye(4)
            self.metrics = {'mean_translation_error_mm': 0, 'mean_rotation_error_deg': 0}
            logger.info("Default calibration data loaded (identity matrices)")
            return
        
        if 'T_C_R' in calibration_data and 'T_T_B' in calibration_data:
            self.T_C_R = np.array(calibration_data['T_C_R']['matrix'], dtype=np.float64)
            self.T_T_B = np.array(calibration_data['T_T_B']['matrix'], dtype=np.float64)
            logger.info("Eye-to-hand calibration matrices loaded successfully")
        else: 
            logger.warning("Required transformation matrices not found in calibration file")
            # Use identity matrices as fallback
            self.T_C_R = np.eye(4)
            self.T_T_B = np.eye(4)
            logger.info("Using identity matrices as fallback")
        
        self.T_R_C = np.linalg.inv(self.T_C_R)
        self.metrics = calibration_data.get('validation_metrics', {'mean_translation_error_mm': 0, 'mean_rotation_error_deg': 0})
        
        # Log calibration quality metrics if available
        if self.metrics.get('mean_translation_error_mm', 0) > 0:
            logger.info(f"Calibration quality - Translation error: {self.metrics['mean_translation_error_mm']:.2f}mm, "
                       f"Rotation error: {self.metrics['mean_rotation_error_deg']:.2f}°")

    def load_workspace_constraints(self):
        """Load robot workspace constraints from constraints.yaml."""
        try:
            # Look for constraints.yaml in the parent directories
            current_dir = Path(__file__).parent
            constraints_path = None
            
            # Try different relative paths to find constraints.yaml
            possible_paths = [
                current_dir / "../config/constraints.yaml",  # monitoring/../config/
                current_dir / "../../config/constraints.yaml",  # monitoring/../../config/
                current_dir / "config/constraints.yaml",  # monitoring/config/
            ]
            
            for path in possible_paths:
                if path.exists():
                    constraints_path = path
                    break
            
            if constraints_path is None:
                raise FileNotFoundError("constraints.yaml not found")
            
            with open(constraints_path, 'r') as f:
                constraints_data = yaml.safe_load(f)
            
            # Extract workspace limits
            workspace = constraints_data.get('workspace', {})
            safety_margins = constraints_data.get('safety_margins', {})
            
            # Apply safety margins if enabled
            margin_enabled = safety_margins.get('enabled', False)
            margin_x = safety_margins.get('margin_x', 0.0) if margin_enabled else 0.0
            margin_y = safety_margins.get('margin_y', 0.0) if margin_enabled else 0.0
            margin_z = safety_margins.get('margin_z', 0.0) if margin_enabled else 0.0
            
            # Set workspace constraints with safety margins applied
            self.workspace_limits = {
                'x_min': workspace.get('x_min', -0.7) + margin_x,
                'x_max': workspace.get('x_max', 0.7) - margin_x,
                'y_min': workspace.get('y_min', -0.7) + margin_y,
                'y_max': workspace.get('y_max', 0.7) - margin_y,
                'z_min': workspace.get('z_min', 0.06) + margin_z,
                'z_max': workspace.get('z_max', 1.1) - margin_z,
            }
            
            # Get robot info
            robot_info = constraints_data.get('robot_info', {})
            self.robot_model = robot_info.get('model', 'RB3-730ES-U')
            
            logger.info(f"Loaded workspace constraints for {self.robot_model}")
            logger.info(f"Workspace: X[{self.workspace_limits['x_min']:.3f}, {self.workspace_limits['x_max']:.3f}], "
                       f"Y[{self.workspace_limits['y_min']:.3f}, {self.workspace_limits['y_max']:.3f}], "
                       f"Z[{self.workspace_limits['z_min']:.3f}, {self.workspace_limits['z_max']:.3f}]")
            
            if margin_enabled:
                logger.info(f"Safety margins applied: X±{margin_x}m, Y±{margin_y}m, Z±{margin_z}m")
                
        except Exception as e:
            logger.warning(f"Could not load workspace constraints: {e}")
            logger.info("Using conservative default workspace limits")
            # Conservative defaults based on typical robot workspace
            self.workspace_limits = {
                'x_min': 0.2, 'x_max': 0.5,
                'y_min': -0.3, 'y_max': 0.3,
                'z_min': 0.2, 'z_max': 0.4,
            }
            self.robot_model = 'RB3-730ES-U'

    def _configure_plots(self):
        self.ax_3d.set_title(f"3. {self.robot_model} Pick & Place Workspace - Press 'Enter' to finalize")
        self.ax_3d.set_xlabel("X (m)"); self.ax_3d.set_ylabel("Y (m)"); self.ax_3d.set_zlabel("Z (m)")
        
        # Use real workspace constraints from constraints.yaml
        x_lim = (self.workspace_limits['x_min'], self.workspace_limits['x_max'])
        y_lim = (self.workspace_limits['y_min'], self.workspace_limits['y_max'])
        z_lim = (self.workspace_limits['z_min'], self.workspace_limits['z_max'])
        
        self.ax_3d.set_xlim(x_lim); self.ax_3d.set_ylim(y_lim); self.ax_3d.set_zlim(z_lim)
        self.ax_3d.set_box_aspect([x_lim[1]-x_lim[0], y_lim[1]-y_lim[0], z_lim[1]-z_lim[0]])
        
        self.ax_xy.set_title("1. Click/Drag to set Pick/Place X-Y Position")
        self.ax_xy.set_xlabel("X (m)"); self.ax_xy.set_ylabel("Y (m)")
        self.ax_xy.set_xlim(x_lim); self.ax_xy.set_ylim(y_lim); self.ax_xy.grid(True)
        self.ax_xy.set_aspect('equal', adjustable='box')

        self.ax_yz.set_title("2. Click/Drag to set Pick/Place Y-Z Height")
        self.ax_yz.set_xlabel("Y (m)"); self.ax_yz.set_ylabel("Z (m)")
        self.ax_yz.set_xlim(y_lim); self.ax_yz.set_ylim(z_lim); self.ax_yz.grid(True)
        self.ax_yz.set_aspect('equal', adjustable='box')
        
    def _create_markers(self):
        # --- Create artists for position preview ---
        self.hint_xy = self.ax_xy.text(0.45, 0, 'Click to set X-Y...', **self._hint_style())
        self.hint_yz = self.ax_yz.text(0, 0.5, 'Click to set Y-Z...', **self._hint_style())
        self.marker_xy, = self.ax_xy.plot([], [], 'ro', ms=10, picker=True, pickradius=5)
        self.marker_yz, = self.ax_yz.plot([], [], 'ro', ms=10, picker=True, pickradius=5)
        self.preview_pos_scatter = self.ax_3d.scatter([], [], [], c='red', s=70, alpha=0.6, ec='black')
        self.vline, = self.ax_xy.plot([], [], **self._line_style())
        self.hline, = self.ax_yz.plot([], [], **self._line_style())
        
        # --- Create artists for orientation preview (3 quivers) ---
        self.preview_orient_quivers = [
            self.ax_3d.quiver([], [], [], [], [], [], color='magenta', arrow_length_ratio=0.2, linewidth=2.5),
            self.ax_3d.quiver([], [], [], [], [], [], color='cyan', arrow_length_ratio=0.2, linewidth=2.5),
            self.ax_3d.quiver([], [], [], [], [], [], color='yellow', arrow_length_ratio=0.2, linewidth=2.5)
        ]
    
    def _hint_style(self):
        """Returns the styling dictionary for hint text."""
        return {'ha': 'center', 'va': 'center', 'style': 'italic', 'color': 'gray'}

    def _line_style(self):
        """Returns the styling dictionary for helper lines."""
        return {'color': 'r', 'lw': 0.8, 'linestyle': '--'}

    def _create_text_boxes(self):
        self.text_boxes = {}
        labels = ['X', 'Y', 'Z']
        positions = [[0.15, 0.13, 0.15, 0.05], [0.42, 0.13, 0.15, 0.05], [0.7, 0.13, 0.15, 0.05]]
        for label, pos in zip(labels, positions):
            ax_box = self.fig.add_axes(pos)
            text_box = TextBox(ax_box, label, initial="", textalignment="center")
            text_box.on_submit(self.submit_text)
            text_box.ax.set_visible(False)
            self.text_boxes[label.lower()] = text_box

    def _create_rotation_sliders(self):
        """Creates the Roll, Pitch, Yaw sliders for orientation control."""
        self.sliders = {}
        labels = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
        keys = ['roll', 'pitch', 'yaw']
        positions = [[0.15, 0.05, 0.2, 0.03], [0.42, 0.05, 0.2, 0.03], [0.7, 0.05, 0.2, 0.03]]
        
        for label, key, pos in zip(labels, keys, positions):
            ax_slider = self.fig.add_axes(pos)
            slider = Slider(ax=ax_slider, label=label, valmin=-180, valmax=180, valinit=0)
            slider.on_changed(self._update_rotation_from_sliders)
            slider.ax.set_visible(False) # Initially hidden
            self.sliders[key] = slider

    def _connect_events(self):
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def on_click(self, event):
        if self.dragged_artist: return
        if event.inaxes and "axes" in event.inaxes.get_label(): return
        
        # Get clicked coordinates
        if event.inaxes == self.ax_xy:
            new_x, new_y = event.xdata, event.ydata
            self.coords['x'], self.coords['y'] = new_x, new_y
        elif event.inaxes == self.ax_yz:
            new_y, new_z = event.xdata, event.ydata
            self.coords['y'], self.coords['z'] = new_y, new_z
        
        # Validate and clamp coordinates to workspace if all are available
        if all(c is not None for c in self.coords.values()):
            x, y, z = self.coords['x'], self.coords['y'], self.coords['z']
            
            # Check if position is outside workspace
            if not self.validate_workspace_position(x, y, z):
                # Show warning and clamp to workspace
                self.show_workspace_warning(x, y, z)
                x_clamped, y_clamped, z_clamped = self.clamp_to_workspace(x, y, z)
                self.coords['x'], self.coords['y'], self.coords['z'] = x_clamped, y_clamped, z_clamped
                
                # Update text boxes if visible
                if hasattr(self, 'text_boxes') and self.text_boxes['x'].ax.get_visible():
                    self.text_boxes['x'].set_val(f"{x_clamped:.3f}")
                    self.text_boxes['y'].set_val(f"{y_clamped:.3f}")
                    self.text_boxes['z'].set_val(f"{z_clamped:.3f}")
        
        # If this is the first click that defines the full position, calculate initial orientation
        if all(c is not None for c in self.coords.values()) and not self.preview_point_visible:
            self._set_initial_orientation()
        
        self._update_pose_preview()

    def on_key_press(self, event):
        if event.key == 'enter' and all(c is not None for c in self.coords.values()):
            self.finalize_pose()
        elif event.key == 'e':
            are_visible = self.text_boxes['x'].ax.get_visible()
            self._toggle_text_boxes(not are_visible)
        elif event.key == 'o':
            self.orientation_mode = "view_aligned" if self.orientation_mode == "downward" else "downward"
            mode_desc = "downward (pick/place)" if self.orientation_mode == "downward" else "view-aligned"
            logger.info(f"Gripper orientation switched to: {mode_desc}")
            # Recalculate initial orientation for the preview if it exists
            if self.preview_point_visible:
                self._set_initial_orientation()
                self._update_pose_preview()
        elif event.key == 'q': plt.close('all')

    def finalize_pose(self):
        """Uses the final position and orientation to generate and store the robot pose."""
        pos = np.array([self.coords['x'], self.coords['y'], self.coords['z']])
        rot_vec = R.from_euler('xyz', [self.orientation_rpy['roll'], self.orientation_rpy['pitch'], self.orientation_rpy['yaw']]).as_rotvec()
        tcp_pose = np.concatenate([pos, rot_vec])
        
        self.robot_poses.append(tcp_pose)
        self._draw_final_pose(tcp_pose, len(self.robot_poses))
        self.print_pose_info(len(self.robot_poses), tcp_pose)
        self._reset_state()
        self.fig.canvas.draw_idle()

    def _update_pose_preview(self):
        """Updates all preview elements: 2D markers, 3D point, and 3D orientation frame."""
        if not all(c is not None for c in self.coords.values()): return

        x, y, z = self.coords['x'], self.coords['y'], self.coords['z']
        
        # Update 2D markers for position
        self.marker_xy.set_data([x], [y]); self.marker_yz.set_data([y], [z])
        self.vline.set_data([x, x], self.ax_xy.get_ylim()); self.hline.set_data(self.ax_yz.get_xlim(), [y, y])
        
        # Update 3D scatter for position
        self.preview_pos_scatter._offsets3d = ([x], [y], [z])
        
        # Update 3D quivers for orientation
        rot_matrix = R.from_euler('xyz', [self.orientation_rpy['roll'], self.orientation_rpy['pitch'], self.orientation_rpy['yaw']]).as_matrix()
        for i in range(3):
            # The quiver plot needs to be reset. A bit of a hack for mplot3d.
            self.preview_orient_quivers[i].set_segments([[[x, y, z], [
                x + self.coordinate_frame_size * rot_matrix[0, i],
                y + self.coordinate_frame_size * rot_matrix[1, i],
                z + self.coordinate_frame_size * rot_matrix[2, i]
            ]]])
        
        if not self.preview_point_visible:
            self.hint_xy.set_text(''); self.hint_yz.set_text('')
            self._toggle_sliders(True) # Show sliders
            self.preview_point_visible = True
        
        self.fig.canvas.draw_idle()

    def _update_rotation_from_sliders(self, val):
        """Callback for when any rotation slider is changed."""
        self.orientation_rpy['roll'] = np.radians(self.sliders['roll'].val)
        self.orientation_rpy['pitch'] = np.radians(self.sliders['pitch'].val)
        self.orientation_rpy['yaw'] = np.radians(self.sliders['yaw'].val)
        self._update_pose_preview()
        
    def _set_initial_orientation(self):
        """Calculates initial orientation and sets slider values."""
        rot_vec = self.calculate_tcp_orientation()
        rpy_rad = R.from_rotvec(rot_vec).as_euler('xyz')
        self.orientation_rpy = {'roll': rpy_rad[0], 'pitch': rpy_rad[1], 'yaw': rpy_rad[2]}
        
        # Update sliders to match the initial orientation
        self.sliders['roll'].set_val(np.degrees(rpy_rad[0]))
        self.sliders['pitch'].set_val(np.degrees(rpy_rad[1]))
        self.sliders['yaw'].set_val(np.degrees(rpy_rad[2]))
        
    def _toggle_sliders(self, visible):
        """Shows or hides the rotation sliders."""
        for slider in self.sliders.values():
            slider.ax.set_visible(visible)
        self.fig.canvas.draw_idle()

    def _reset_state(self):
        self.coords = {'x': None, 'y': None, 'z': None}
        self.preview_point_visible = False
        self.marker_xy.set_data([], []); self.marker_yz.set_data([], [])
        self.vline.set_data([], []); self.hline.set_data([], [])
        self.preview_pos_scatter._offsets3d = ([], [], [])
        # Reset and hide orientation preview
        for q in self.preview_orient_quivers: q.set_segments([])
        self._toggle_text_boxes(False)
        self._toggle_sliders(False)
        self.hint_xy.set_text('Click to set X-Y...'); self.hint_yz.set_text('Click to set Y-Z...')
    
    # --- Other methods (mostly unchanged) ---
    def on_pick(self, event): self.dragged_artist = event.artist
    def on_release(self, event): self.dragged_artist = None
    def on_motion(self, event):
        if self.dragged_artist is None or not event.inaxes: return
        if self.dragged_artist == self.marker_xy: self.coords['x'], self.coords['y'] = event.xdata, event.ydata
        elif self.dragged_artist == self.marker_yz: self.coords['y'], self.coords['z'] = event.xdata, event.ydata
        self._update_pose_preview()

    def submit_text(self, text):
        try:
            value = float(text)
            # Identify which text box was updated
            updated_key = None
            for key, tb in self.text_boxes.items():
                if tb.ax.is_figure_set() and tb.ax.get_visible() and tb.text == text:
                    updated_key = key
                    break
            
            if updated_key:
                # Store old coordinates
                old_coords = self.coords.copy()
                self.coords[updated_key] = value
                
                # Validate workspace if all coordinates are available
                if all(c is not None for c in self.coords.values()):
                    x, y, z = self.coords['x'], self.coords['y'], self.coords['z']
                    
                    if not self.validate_workspace_position(x, y, z):
                        # Show warning and clamp to workspace
                        self.show_workspace_warning(x, y, z)
                        x_clamped, y_clamped, z_clamped = self.clamp_to_workspace(x, y, z)
                        self.coords['x'], self.coords['y'], self.coords['z'] = x_clamped, y_clamped, z_clamped
                        
                        # Update text boxes with clamped values
                        self.text_boxes['x'].set_val(f"{x_clamped:.3f}")
                        self.text_boxes['y'].set_val(f"{y_clamped:.3f}")
                        self.text_boxes['z'].set_val(f"{z_clamped:.3f}")
                
                self._update_pose_preview()
        except (ValueError, TypeError): 
            logger.warning(f"Invalid input: '{text}'. Please enter a valid number.")

    def _toggle_text_boxes(self, visible):
        if visible and all(c is not None for c in self.coords.values()):
            self.text_boxes['x'].set_val(f"{self.coords['x']:.4f}")
            self.text_boxes['y'].set_val(f"{self.coords['y']:.4f}")
            self.text_boxes['z'].set_val(f"{self.coords['z']:.4f}")
        for tb in self.text_boxes.values(): tb.ax.set_visible(visible)
        self.fig.canvas.draw_idle()

    def _draw_static_elements(self):
        self._draw_coordinate_frame(np.eye(4), "Robot Base", ['red', 'green', 'blue'])
        self._draw_coordinate_frame(self.T_R_C, "Camera", ['darkred', 'darkgreen', 'darkblue'])
        self._draw_workspace_boundary()
        self.fig.canvas.draw_idle()

    def _create_legend(self):
        from matplotlib.lines import Line2D
        elements = [Line2D([0], [0], color='magenta', lw=4, label='Gripper X-axis'),
                    Line2D([0], [0], color='cyan', lw=4, label='Gripper Y-axis'),
                    Line2D([0], [0], color='yellow', lw=4, label='Gripper Z-axis (downward)')]
        self.ax_3d.legend(handles=elements, title="Pick & Place Gripper", loc='upper left')

    def validate_workspace_position(self, x, y, z):
        """Validate if position is within robot workspace constraints."""
        x_valid = self.workspace_limits['x_min'] <= x <= self.workspace_limits['x_max']
        y_valid = self.workspace_limits['y_min'] <= y <= self.workspace_limits['y_max']
        z_valid = self.workspace_limits['z_min'] <= z <= self.workspace_limits['z_max']
        return x_valid and y_valid and z_valid
    
    def clamp_to_workspace(self, x, y, z):
        """Clamp coordinates to workspace limits."""
        x_clamped = np.clip(x, self.workspace_limits['x_min'], self.workspace_limits['x_max'])
        y_clamped = np.clip(y, self.workspace_limits['y_min'], self.workspace_limits['y_max'])
        z_clamped = np.clip(z, self.workspace_limits['z_min'], self.workspace_limits['z_max'])
        return x_clamped, y_clamped, z_clamped
    
    def show_workspace_warning(self, x, y, z):
        """Show warning when attempting to set position outside workspace."""
        violations = []
        if x < self.workspace_limits['x_min'] or x > self.workspace_limits['x_max']:
            violations.append(f"X: {x:.3f}m (limits: {self.workspace_limits['x_min']:.3f} to {self.workspace_limits['x_max']:.3f}m)")
        if y < self.workspace_limits['y_min'] or y > self.workspace_limits['y_max']:
            violations.append(f"Y: {y:.3f}m (limits: {self.workspace_limits['y_min']:.3f} to {self.workspace_limits['y_max']:.3f}m)")
        if z < self.workspace_limits['z_min'] or z > self.workspace_limits['z_max']:
            violations.append(f"Z: {z:.3f}m (limits: {self.workspace_limits['z_min']:.3f} to {self.workspace_limits['z_max']:.3f}m)")
        
        if violations:
            warning_msg = f"⚠️ Position clamped to workspace limits:\n" + "\n".join(violations)
            logger.warning(warning_msg)
            # Update plot title to show warning temporarily
            self.ax_3d.set_title(f"⚠️ Position constrained to {self.robot_model} workspace")
            plt.pause(0.1)  # Brief pause to show warning

    def _draw_coordinate_frame(self, T, name, colors):
        origin = T[:3, 3]
        for i, color in enumerate(colors):
            end = origin + self.coordinate_frame_size * T[:3, i]
            self.ax_3d.quiver(origin[0],origin[1],origin[2], end[0]-origin[0],end[1]-origin[1],end[2]-origin[2],
                              color=color, arrow_length_ratio=0.2, linewidth=2)
        self.ax_3d.text(origin[0], origin[1], origin[2] + self.coordinate_frame_size, name, fontsize=10)

    def _draw_final_pose(self, pose, pose_number):
        T_tcp = np.eye(4)
        T_tcp[:3, 3] = pose[:3]
        T_tcp[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
        self.ax_3d.scatter(pose[0], pose[1], pose[2], c='blue', s=80, ec='black', depthshade=True)
        self._draw_coordinate_frame(T_tcp, f"Grip{pose_number}", ['magenta', 'cyan', 'yellow'])

    def _draw_workspace_boundary(self):
        """Draw workspace boundary box using robot workspace constraints."""
        # Use actual workspace limits instead of axis limits
        x_lim = (self.workspace_limits['x_min'], self.workspace_limits['x_max'])
        y_lim = (self.workspace_limits['y_min'], self.workspace_limits['y_max'])
        z_lim = (self.workspace_limits['z_min'], self.workspace_limits['z_max'])
        
        # Define the 8 corners of the workspace boundary box
        corners = np.array([
            [x_lim[0], y_lim[0], z_lim[0]],  # 0
            [x_lim[1], y_lim[0], z_lim[0]],  # 1
            [x_lim[1], y_lim[1], z_lim[0]],  # 2
            [x_lim[0], y_lim[1], z_lim[0]],  # 3
            [x_lim[0], y_lim[0], z_lim[1]],  # 4
            [x_lim[1], y_lim[0], z_lim[1]],  # 5
            [x_lim[1], y_lim[1], z_lim[1]],  # 6
            [x_lim[0], y_lim[1], z_lim[1]],  # 7
        ])
        
        # Define the 12 edges that connect the corners
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom edges
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top edges
            [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
        ]
        
        # Plot each edge with more prominent styling
        for edge in edges:
            start, end = corners[edge[0]], corners[edge[1]]
            self.ax_3d.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                            'r-', alpha=0.6, linewidth=1.5, label='Workspace Boundary' if edge == [0, 1] else "")
        
        # Add workspace annotation
        center_x = (x_lim[0] + x_lim[1]) / 2
        center_y = (y_lim[0] + y_lim[1]) / 2
        center_z = z_lim[1]
        self.ax_3d.text(center_x, center_y, center_z + 0.05, 
                       f"{self.robot_model}\nWorkspace", fontsize=9, ha='center', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    def calculate_tcp_orientation(self):
        if self.orientation_mode == "view_aligned":
            elev, azim = np.radians(self.ax_3d.elev), np.radians(self.ax_3d.azim)
            v_dir = np.array([np.cos(elev)*np.cos(azim), np.cos(elev)*np.sin(azim), -np.sin(elev)])
            v_up = np.array([-np.sin(elev)*np.cos(azim), -np.sin(elev)*np.sin(azim), -np.cos(elev)])
            z_ax = v_dir / np.linalg.norm(v_dir)
            y_ax = v_up - np.dot(v_up, z_ax) * z_ax
            y_ax /= np.linalg.norm(y_ax)
            x_ax = np.cross(y_ax, z_ax)
            return R.from_matrix(np.column_stack([x_ax, y_ax, z_ax])).as_rotvec()
        else: return np.array([np.pi, 0, 0])

    def print_pose_info(self, pose_number, tcp_pose):
        mode_desc = "Pick/Place (downward)" if self.orientation_mode == "downward" else "View-aligned"
        print(f"\n--- Gripper Pose #{pose_number} [{mode_desc}] ---")
        print(f"Position (mm): [X={tcp_pose[0]*1000:.1f}, Y={tcp_pose[1]*1000:.1f}, Z={tcp_pose[2]*1000:.1f}]")
        rpy_deg = np.degrees(R.from_rotvec(tcp_pose[3:]).as_euler('xyz'))
        print(f"Rotation (deg): [Roll={rpy_deg[0]:.1f}, Pitch={rpy_deg[1]:.1f}, Yaw={rpy_deg[2]:.1f}]")
        
    def export_poses(self, filename: str = "pick_place_poses.json"):
        if not self.robot_poses: logger.warning("No poses to export."); return
        data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), 
            "calibration_file": str(self.calibration_path),
            "application": "pick_and_place",
            "robot_model": self.robot_model,
            "workspace_constraints": self.workspace_limits,
            "poses": []
        }
        for i, pose in enumerate(self.robot_poses):
            rpy_deg = np.degrees(R.from_rotvec(pose[3:]).as_euler('xyz'))
            data["poses"].append({
                "pose_number": i+1, 
                "gripper_position_mm": [round(p,1) for p in (pose[:3]*1000).tolist()],
                "gripper_rotation_rpy_deg": [round(r,1) for r in rpy_deg],
                "orientation_mode": "downward_pick_place"
            })
        with open(filename, 'w') as f: json.dump(data, f, indent=2)
        logger.info(f"Exported {len(self.robot_poses)} pick & place poses to {filename}")
    
    def get_poses(self):
        """
        Get the list of captured robot poses.
        
        Returns:
            list: List of TCP poses as [x, y, z, rx, ry, rz] arrays
        """
        return self.robot_poses.copy()
    
    def clear_poses(self):
        """Clear all captured poses."""
        self.robot_poses.clear()
        logger.info("All poses cleared")
    
    def run(self):
        print("\n" + "="*60 + "\nADVANCED 3D INTERACTIVE ROBOT POSE CONTROL\n" + "="*60 + "\nCONTROLS:")
        print("  • Click/Drag on 2D plots to set POSITION.")
        print("  • Use sliders at the bottom to set ORIENTATION.")
        print("  • Press 'e' to show/hide text boxes for precise position.")
        print("  • Press 'Enter' to finalize the pose.")
        print("  • Press 'o' to toggle initial orientation mode.")
        print("  • Press 'q' to quit.\n" + "="*60)
        try: plt.show()
        except KeyboardInterrupt: print("\nExiting...")
        finally:
            if self.robot_poses: self.export_poses()

def main():
    try:
        visualizer = AdvancedPoseVisualizer()
        visualizer.run()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()