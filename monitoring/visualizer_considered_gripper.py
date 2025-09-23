#!/usr/bin/env python3
"""
Advanced 3D Interactive Pose Control Visualizer

This tool combines a multi-view 2D plotting interface for precise point definition
with the back-end logic for generating TCP poses, now with rotational adjustment.

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
        
        # Gripper mode settings
        self.gripper_mode = False  # False = TCP mode, True = Gripper mode
        self.gripper_offset = 0.085  # 85mm gripper offset in meters

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
        self._create_gripper_mode_toggle() # NEW: Gripper mode control
        self._connect_events()
        self._draw_static_elements()
        self._create_legend()
        
        # Initialize plot titles
        self._update_plot_titles()
        
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
            
            # Add reachability constraints based on empirically validated robot capabilities
            # Updated based on real robot testing: 600mm reliable with downward gripper
            # Zones adjusted to reflect actual operational capabilities
            
            self.reachability_limits = {
                'max_radius_mm': 650,        # Conservative maximum with small headroom
                'warning_radius_mm': 620,    # Extended reach limit (approaching maximum)
                'safe_radius_mm': 600,       # Reliable reach validated with downward gripper
            }
            
            # Get robot info
            robot_info = constraints_data.get('robot_info', {})
            self.robot_model = robot_info.get('model', 'RB3-730ES-U')
            
            logger.info(f"Loaded workspace constraints for {self.robot_model}")
            logger.info(f"Workspace: X[{self.workspace_limits['x_min']:.3f}, {self.workspace_limits['x_max']:.3f}], Y[{self.workspace_limits['y_min']:.3f}, {self.workspace_limits['y_max']:.3f}], Z[{self.workspace_limits['z_min']:.3f}, {self.workspace_limits['z_max']:.3f}]")
            logger.info(f"Reachability: Safe≤{self.reachability_limits['safe_radius_mm']}mm, Warning≤{self.reachability_limits['warning_radius_mm']}mm, Max≤{self.reachability_limits['max_radius_mm']}mm")
            
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
            # Default reachability limits based on URDF (conservative defaults)
            self.reachability_limits = {
                'max_radius_mm': 600.0,  # Conservative for unknown robot
                'warning_radius_mm': 500.0,
                'safe_radius_mm': 400.0,
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
        self._add_reachable_zones_2d()

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
            slider = Slider(ax_slider, label, -180, 180, valinit=0, valfmt='%0.1f°')
            slider.on_changed(self._update_rotation_from_sliders)
            self.sliders[key] = slider
            ax_slider.set_visible(False)  # Start hidden

    def _create_gripper_mode_toggle(self):
        """Create gripper/TCP mode toggle button."""
        from matplotlib.widgets import Button
        
        # Position button in bottom right area
        ax_button = self.fig.add_axes([0.85, 0.02, 0.12, 0.04])
        self.gripper_button = Button(ax_button, 'TCP Mode')
        self.gripper_button.on_clicked(self._toggle_gripper_mode)
        
        # Style the button
        ax_button.set_facecolor('lightblue')
        
        # Create mode indicator text
        self.mode_text = self.fig.text(0.85, 0.08, 'Mode: TCP (Tool Center Point)', 
                                      fontsize=10, ha='left', va='bottom',
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))

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
            
            # Check if position is outside workspace or unreachable
            validation = self.validate_position_complete(x, y, z)
            if not validation['overall_valid']:
                # Show warning and clamp to workspace/reachability constraints
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
        elif event.key == 'g':  # NEW: 'g' key to toggle gripper mode
            self._toggle_gripper_mode()
        elif event.key == 'q': plt.close('all')

    def finalize_pose(self):
        """Uses the final position and orientation to generate and store the robot pose."""
        pos = np.array([self.coords['x'], self.coords['y'], self.coords['z']])
        rot_vec = R.from_euler('xyz', [self.orientation_rpy['roll'], self.orientation_rpy['pitch'], self.orientation_rpy['yaw']]).as_rotvec()
        
        # Store gripper coordinates for export
        if self.gripper_mode:
            self._last_gripper_coords = pos.copy()
        
        # Convert gripper coordinates to TCP coordinates if in gripper mode
        if self.gripper_mode:
            tcp_pose = self._convert_gripper_to_tcp(pos, rot_vec)
        else:
            tcp_pose = np.concatenate([pos, rot_vec])
        
        self.robot_poses.append(tcp_pose)
        self._draw_final_pose(tcp_pose, len(self.robot_poses))
        self.print_pose_info(len(self.robot_poses), tcp_pose)
        self._reset_state()
        self.fig.canvas.draw_idle()
    
    def _convert_gripper_to_tcp(self, gripper_pos, rot_vec):
        """Convert gripper position to TCP position by applying offset."""
        # Create rotation matrix from rotation vector
        rotation_matrix = R.from_rotvec(rot_vec).as_matrix()
        
        # Apply offset along the Z-axis of the gripper (TCP is 85mm behind gripper tip)
        # In gripper frame: Z-axis points downward, so TCP is +Z from gripper tip
        gripper_z_axis = rotation_matrix[:, 2]  # Z-axis of gripper frame
        tcp_position = gripper_pos + self.gripper_offset * gripper_z_axis
        
        return np.concatenate([tcp_position, rot_vec])

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
        
    def _toggle_gripper_mode(self, event=None):
        """Toggle between TCP and Gripper modes."""
        self.gripper_mode = not self.gripper_mode
        
        if self.gripper_mode:
            self.gripper_button.label.set_text('Gripper Mode')
            self.gripper_button.ax.set_facecolor('lightgreen')
            self.mode_text.set_text('Mode: GRIPPER (Gripper Tip Position)\nAutomatic 85mm offset applied')
            mode_type = "gripper tip positioning"
        else:
            self.gripper_button.label.set_text('TCP Mode')
            self.gripper_button.ax.set_facecolor('lightblue')
            self.mode_text.set_text('Mode: TCP (Tool Center Point)')
            mode_type = "TCP positioning"
        
        logger.info(f"Switched to {mode_type} mode")
        
        # Update plot titles to reflect current mode
        self._update_plot_titles()
        
        # If preview is visible, update it with new coordinates
        if self.preview_point_visible:
            self._update_pose_preview()
        
        self.fig.canvas.draw_idle()
    
    def _update_plot_titles(self):
        """Update plot titles based on current mode."""
        mode_suffix = " (Gripper)" if self.gripper_mode else " (TCP)"
        
        self.ax_3d.set_title(f"3. {self.robot_model} Workspace{mode_suffix} - Press 'Enter' to finalize")
        self.ax_xy.set_title(f"1. Click/Drag to set X-Y Position{mode_suffix}")
        self.ax_yz.set_title(f"2. Click/Drag to set Y-Z Position{mode_suffix}")
        
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
                if tb.ax.get_visible() and tb.text == text:
                    updated_key = key
                    break
            
            if updated_key:
                # Store old coordinates
                old_coords = self.coords.copy()
                self.coords[updated_key] = value
                
                # Validate workspace and reachability if all coordinates are available
                if all(c is not None for c in self.coords.values()):
                    x, y, z = self.coords['x'], self.coords['y'], self.coords['z']
                    
                    validation = self.validate_position_complete(x, y, z)
                    if not validation['overall_valid']:
                        # Show warning and clamp to workspace/reachability constraints
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
    
    def validate_reachability(self, x, y, z):
        """Validate if position is reachable by robot arm."""
        # Convert to mm and calculate distance from robot base
        distance_mm = np.sqrt((x * 1000)**2 + (y * 1000)**2)
        
        # Check against reachability limits
        is_reachable = distance_mm <= self.reachability_limits['max_radius_mm']
        is_safe = distance_mm <= self.reachability_limits['safe_radius_mm']
        is_warning = distance_mm <= self.reachability_limits['warning_radius_mm']
        
        return {
            'reachable': is_reachable,
            'safe': is_safe,
            'warning_zone': not is_warning and is_reachable,
            'distance_mm': distance_mm
        }
    
    def validate_position_complete(self, x, y, z):
        """Complete validation including workspace and reachability."""
        workspace_valid = self.validate_workspace_position(x, y, z)
        reachability_info = self.validate_reachability(x, y, z)
        
        return {
            'workspace_valid': workspace_valid,
            'reachable': reachability_info['reachable'],
            'safe': reachability_info['safe'],
            'warning_zone': reachability_info['warning_zone'],
            'distance_mm': reachability_info['distance_mm'],
            'overall_valid': workspace_valid and reachability_info['reachable']
        }
    
    def clamp_to_workspace(self, x, y, z):
        """Clamp coordinates to workspace limits and reachability constraints."""
        # First clamp to workspace bounds
        x_clamped = np.clip(x, self.workspace_limits['x_min'], self.workspace_limits['x_max'])
        y_clamped = np.clip(y, self.workspace_limits['y_min'], self.workspace_limits['y_max'])
        z_clamped = np.clip(z, self.workspace_limits['z_min'], self.workspace_limits['z_max'])
        
        # Then check reachability and clamp if needed
        distance_mm = np.sqrt((x_clamped * 1000)**2 + (y_clamped * 1000)**2)
        if distance_mm > self.reachability_limits['max_radius_mm']:
            # Scale down to maximum reachable radius
            scale_factor = self.reachability_limits['max_radius_mm'] / distance_mm
            x_clamped *= scale_factor
            y_clamped *= scale_factor
            
        return x_clamped, y_clamped, z_clamped
    
    def show_workspace_warning(self, x, y, z):
        """Show warning when attempting to set position outside workspace or reachability."""
        validation = self.validate_position_complete(x, y, z)
        
        violations = []
        
        # Check workspace violations
        if not validation['workspace_valid']:
            if x < self.workspace_limits['x_min'] or x > self.workspace_limits['x_max']:
                violations.append(f"X: {x:.3f}m (limits: {self.workspace_limits['x_min']:.3f} to {self.workspace_limits['x_max']:.3f}m)")
            if y < self.workspace_limits['y_min'] or y > self.workspace_limits['y_max']:
                violations.append(f"Y: {y:.3f}m (limits: {self.workspace_limits['y_min']:.3f} to {self.workspace_limits['y_max']:.3f}m)")
            if z < self.workspace_limits['z_min'] or z > self.workspace_limits['z_max']:
                violations.append(f"Z: {z:.3f}m (limits: {self.workspace_limits['z_min']:.3f} to {self.workspace_limits['z_max']:.3f}m)")
        
        # Check reachability violations
        if not validation['reachable']:
            violations.append(f"Distance: {validation['distance_mm']:.1f}mm (max reachable: {self.reachability_limits['max_radius_mm']:.1f}mm)")
        elif validation['warning_zone']:
            violations.append(f"Warning: {validation['distance_mm']:.1f}mm from base (recommended ≤{self.reachability_limits['warning_radius_mm']:.1f}mm)")
        
        if violations:
            warning_msg = f"WARNING: Position adjusted for robot constraints:\n" + "\n".join(violations)
            logger.warning(warning_msg)
            
            # Update plot title based on constraint type
            if not validation['reachable']:
                self.ax_3d.set_title(f"WARNING: Position limited to {self.robot_model} reachable zone")
            elif validation['warning_zone']:
                self.ax_3d.set_title(f"WARNING: Position in {self.robot_model} warning zone - use caution")
            else:
                self.ax_3d.set_title(f"WARNING: Position constrained to {self.robot_model} workspace")
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
        """Draw workspace boundary box and reachable zone indicators using robot workspace constraints."""
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
        
        # Draw reachable zone indicators
        self._draw_reachable_zones()
    
    def _draw_reachable_zones(self):
        """Draw circular indicators showing robot reachable zones in XY plane."""
        # Draw zones at the middle Z height for visibility
        z_mid = (self.workspace_limits['z_min'] + self.workspace_limits['z_max']) / 2
        
        # Convert radius from mm to meters for plotting
        max_radius = self.reachability_limits['max_radius_mm'] / 1000.0
        warning_radius = self.reachability_limits['warning_radius_mm'] / 1000.0
        safe_radius = self.reachability_limits['safe_radius_mm'] / 1000.0
        
        # Create circle points
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Draw safe zone (green)
        x_safe = safe_radius * np.cos(theta)
        y_safe = safe_radius * np.sin(theta)
        z_safe = np.full_like(x_safe, z_mid)
        self.ax_3d.plot(x_safe, y_safe, z_safe, 'g-', alpha=0.7, linewidth=2, 
                       label=f'Safe Zone (≤{self.reachability_limits["safe_radius_mm"]:.0f}mm)')
        
        # Draw warning zone (yellow)
        x_warning = warning_radius * np.cos(theta)
        y_warning = warning_radius * np.sin(theta)
        z_warning = np.full_like(x_warning, z_mid)
        self.ax_3d.plot(x_warning, y_warning, z_warning, 'y-', alpha=0.7, linewidth=2,
                       label=f'Warning Zone (≤{self.reachability_limits["warning_radius_mm"]:.0f}mm)')
        
        # Draw maximum reachable zone (red)
        x_max = max_radius * np.cos(theta)
        y_max = max_radius * np.sin(theta)
        z_max = np.full_like(x_max, z_mid)
        self.ax_3d.plot(x_max, y_max, z_max, 'r-', alpha=0.7, linewidth=2,
                       label=f'Max Reach (≤{self.reachability_limits["max_radius_mm"]:.0f}mm)')
        
        # Add zone legend and annotations
        self.ax_3d.text(0, safe_radius + 0.05, z_mid, 'SAFE', fontsize=8, ha='center', 
                       color='green', weight='bold')
        self.ax_3d.text(0, warning_radius + 0.05, z_mid, 'CAUTION', fontsize=8, ha='center', 
                       color='orange', weight='bold')
        self.ax_3d.text(0, max_radius + 0.05, z_mid, 'MAX', fontsize=8, ha='center', 
                       color='red', weight='bold')
        
        # Add robot base marker at origin
        self.ax_3d.scatter([0], [0], [z_mid], c='black', s=100, marker='s', 
                          label='Robot Base', alpha=0.8)
    
    def _add_reachable_zones_2d(self):
        """Add reachable zone circles to the XY 2D plot."""
        # Convert radius from mm to meters for plotting
        max_radius = self.reachability_limits['max_radius_mm'] / 1000.0
        warning_radius = self.reachability_limits['warning_radius_mm'] / 1000.0
        safe_radius = self.reachability_limits['safe_radius_mm'] / 1000.0
        
        # Create circle points
        theta = np.linspace(0, 2*np.pi, 100)
        
        # Draw circles on XY plot
        # Safe zone (green)
        x_safe = safe_radius * np.cos(theta)
        y_safe = safe_radius * np.sin(theta)
        self.ax_xy.plot(x_safe, y_safe, 'g-', alpha=0.5, linewidth=1.5)
        self.ax_xy.fill(x_safe, y_safe, color='green', alpha=0.1)
        
        # Warning zone (yellow)
        x_warning = warning_radius * np.cos(theta)
        y_warning = warning_radius * np.sin(theta)
        self.ax_xy.plot(x_warning, y_warning, 'y-', alpha=0.5, linewidth=1.5)
        self.ax_xy.fill(x_warning, y_warning, color='yellow', alpha=0.1)
        
        # Maximum reachable zone (red)
        x_max = max_radius * np.cos(theta)
        y_max = max_radius * np.sin(theta)
        self.ax_xy.plot(x_max, y_max, 'r-', alpha=0.5, linewidth=1.5)
        
        # Add robot base marker at origin
        self.ax_xy.scatter([0], [0], c='black', s=80, marker='s', alpha=0.8, zorder=10)
        
        # Add zone labels
        self.ax_xy.text(0, safe_radius - 0.05, 'SAFE', fontsize=7, ha='center', 
                       color='green', weight='bold', alpha=0.8)
        self.ax_xy.text(0, warning_radius - 0.05, 'CAUTION', fontsize=7, ha='center', 
                       color='orange', weight='bold', alpha=0.8)
        self.ax_xy.text(0, max_radius - 0.05, 'MAX', fontsize=7, ha='center', 
                       color='red', weight='bold', alpha=0.8)
    
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
        coord_mode = "Gripper Mode" if self.gripper_mode else "TCP Mode"
        
        print(f"\n--- Pose #{pose_number} [{mode_desc}, {coord_mode}] ---")
        
        if self.gripper_mode:
            # Show both gripper coordinates (input) and TCP coordinates (output)
            gripper_pos = np.array([self.coords['x'], self.coords['y'], self.coords['z']])
            print(f"Gripper Position (mm): [X={gripper_pos[0]*1000:.1f}, Y={gripper_pos[1]*1000:.1f}, Z={gripper_pos[2]*1000:.1f}]")
            print(f"TCP Position (mm):     [X={tcp_pose[0]*1000:.1f}, Y={tcp_pose[1]*1000:.1f}, Z={tcp_pose[2]*1000:.1f}] (auto-calculated)")
        else:
            print(f"TCP Position (mm): [X={tcp_pose[0]*1000:.1f}, Y={tcp_pose[1]*1000:.1f}, Z={tcp_pose[2]*1000:.1f}]")
        
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
            "coordinate_mode": "gripper" if self.gripper_mode else "tcp",
            "gripper_offset_mm": self.gripper_offset * 1000 if self.gripper_mode else 0,
            "poses": []
        }
        for i, pose in enumerate(self.robot_poses):
            rpy_deg = np.degrees(R.from_rotvec(pose[3:]).as_euler('xyz'))
            pose_data = {
                "pose_number": i+1, 
                "tcp_position_mm": [round(p,1) for p in (pose[:3]*1000).tolist()],
                "tcp_rotation_rpy_deg": [round(r,1) for r in rpy_deg],
                "orientation_mode": self.orientation_mode,
                "coordinate_mode": "gripper" if self.gripper_mode else "tcp"
            }
            
            # Add gripper coordinates if in gripper mode
            if self.gripper_mode and hasattr(self, '_last_gripper_coords'):
                pose_data["gripper_position_mm"] = [round(p,1) for p in (self._last_gripper_coords * 1000).tolist()]
            
            data["poses"].append(pose_data)
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
        print("\n" + "="*70 + "\nADVANCED 3D INTERACTIVE ROBOT POSE CONTROL\n" + "="*70 + "\nCONTROLS:")
        print("  • Click/Drag on 2D plots to set POSITION.")
        print("  • Use sliders at the bottom to set ORIENTATION.")
        print("  • Press 'e' to show/hide text boxes for precise position.")
        print("  • Press 'g' to toggle Gripper/TCP mode.")
        print("  • Click button or press 'g' to switch between modes.")
        print("  • Press 'Enter' to finalize the pose.")
        print("  • Press 'o' to toggle initial orientation mode.")
        print("  • Press 'q' to quit.")
        print("\nMODES:")
        print("  TCP Mode: Position robot Tool Center Point directly")
        print("  Gripper Mode: Position gripper tip (85mm offset auto-applied)")
        print("="*70)
        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            if self.robot_poses:
                self.export_poses()

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