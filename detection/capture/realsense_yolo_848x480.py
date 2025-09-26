import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs

# Import our clean detection classes
from .yolo_detector import YOLODetector, Detection
from .sam2_segmentor import SAM2Segmentor

# =============================================================================
# CONFIGURATION - Modify these variables instead of using command line args
# =============================================================================

# Get the detection module directory (2 levels up from this file)
DETECTION_MODULE_DIR = Path(__file__).parent.parent
MODELS_DIR = DETECTION_MODULE_DIR / "models"

# Dataset capture settings
DATASET_DIR = DETECTION_MODULE_DIR / "captures"  # Where to save RGB/Depth/Mask data
AUTO_CAPTURE = True  # True = auto capture, False = press 'c' to capture
CAPTURE_INTERVAL = 3.0  # Seconds between auto captures
OBJECT_LABEL = "cup"  # Label for captured objects

# YOLO model settings
ENABLE_YOLO = True  # Set to False to disable object detection
YOLO_MODEL_PATH = MODELS_DIR / "yolo" / "yolov8n.pt"  # Path to YOLO model
YOLO_CONFIDENCE = 0.5  # Minimum confidence threshold
YOLO_TARGET_CLASSES = [41]  # Filter for specific classes (41 = cup)

# SAM2 settings (optional - set to None to disable)
SAM2_CONFIG = MODELS_DIR / "sam2" / "configs" / "sam2_hiera_tiny.yaml"
SAM2_CHECKPOINT = MODELS_DIR / "sam2" / "checkpoints" / "sam2_hiera_tiny.pth"
SAM2_DEVICE = "cuda"  # cuda, cuda:0, or cpu

# To disable SAM2, uncomment these lines:
# SAM2_CONFIG = None
# SAM2_CHECKPOINT = None

# =============================================================================


def ensure_dataset_dirs(dataset_dir: Path) -> None:
    """Create dataset directory structure."""
    for sub in ("rgb", "depth", "mask", "meta"):
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)


def save_capture(
    dataset_dir: Path,
    frame_id: str,
    color_image: np.ndarray,
    depth_image: np.ndarray,
    mask: np.ndarray,
    intrinsics: rs.intrinsics,
    depth_scale: float,
    detections: Optional[list] = None,
) -> None:
    """Save RGB, depth, mask and metadata to LINEMOD format."""
    rgb_path = dataset_dir / "rgb" / f"{frame_id}.png"
    depth_path = dataset_dir / "depth" / f"{frame_id}.png"
    mask_path = dataset_dir / "mask" / f"{frame_id}.png"
    meta_path = dataset_dir / "meta" / f"{frame_id}.json"

    cv2.imwrite(str(rgb_path), color_image)

    depth_mm = (depth_image.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16)
    cv2.imwrite(str(depth_path), depth_mm)

    cv2.imwrite(str(mask_path), mask)

    metadata = {
        "label": OBJECT_LABEL,
        "timestamp": frame_id,
        "intrinsics": {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
        },
        "depth_scale_m": depth_scale,
        "detections": [det.__dict__ for det in detections] if detections else [],
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    print("=== RealSense D456 + YOLO + SAM2 Capture System ===")
    print(f"Target objects: CUP/MUG detection (YOLO class 41)")
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Auto-capture: {AUTO_CAPTURE} ({'every ' + str(CAPTURE_INTERVAL) + 's' if AUTO_CAPTURE else 'press c'})")
    print(f"Object label: {OBJECT_LABEL}")
    print(f"Models directory: {MODELS_DIR}")
    print()

    # Check if models directory exists
    if not MODELS_DIR.exists():
        print(f"⚠️  Models directory not found: {MODELS_DIR}")
        print("Consider running: python -m detection.models.model_manager --setup-basic")
        print()

    # Initialize detection models
    yolo_detector = None
    if ENABLE_YOLO:
        yolo_detector = YOLODetector(YOLO_MODEL_PATH, YOLO_CONFIDENCE, YOLO_TARGET_CLASSES)
        if not yolo_detector.is_available():
            print("⚠️  YOLO detection will be disabled")
            yolo_detector = None

    # Initialize SAM2 segmentor
    sam2_segmentor = SAM2Segmentor(SAM2_CONFIG, SAM2_CHECKPOINT, SAM2_DEVICE)
    if not sam2_segmentor.is_available():
        print("⚠️  SAM2 segmentation will use bounding box fallback")

    print(f"YOLO enabled: {yolo_detector is not None}")
    print(f"SAM2 enabled: {sam2_segmentor.is_available()}")
    print()

    # Setup RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intrinsics = color_stream.get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        align = rs.align(rs.stream.color)

        print("RealSense D456 streaming at 848x480...")
        print("Controls:")
        print("  'c' - Capture frame")
        print("  'd' - Toggle detection overlay")  
        print("  'q' - Quit")
        print()

        # Ensure dataset directory exists
        ensure_dataset_dirs(DATASET_DIR)

        last_capture_time = datetime.utcnow()
        show_detections = True
        frame_count = 0

        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Run YOLO detection
            detections = []
            if yolo_detector and yolo_detector.is_available():
                detections = yolo_detector.detect(color_image)

            # Create display image
            display_image = color_image.copy()
            
            # Draw detections if enabled
            if show_detections and detections:
                for det in detections:
                    # Draw bounding box
                    cv2.rectangle(display_image, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
                    # Draw label and confidence
                    label_text = f"{det.label}: {det.confidence:.2f}"
                    cv2.putText(display_image, label_text, (det.x1, det.y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Create depth colormap for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET,
            )

            # Combine color and depth for display
            combined_display = np.hstack((display_image, depth_colormap))
            
            # Add status text
            status_text = f"Frame: {frame_count} | Detections: {len(detections)} | Auto: {AUTO_CAPTURE}"
            cv2.putText(combined_display, status_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("RealSense + YOLO + SAM2", combined_display)

            key = cv2.waitKey(1) & 0xFF
            now = datetime.utcnow()

            # Check for capture trigger
            should_capture = False
            if AUTO_CAPTURE:
                elapsed = (now - last_capture_time).total_seconds()
                should_capture = elapsed >= max(CAPTURE_INTERVAL, 0.1)
            elif key == ord("c"):
                should_capture = True

            # Capture frame if triggered
            if should_capture:
                frame_id = now.strftime("%Y%m%d_%H%M%S_%f")
                
                # Generate mask using SAM2 or fallback to bounding boxes
                mask = sam2_segmentor.generate_mask_from_detections(color_image, detections)
                
                # Save the capture
                save_capture(
                    dataset_dir=DATASET_DIR,
                    frame_id=frame_id,
                    color_image=color_image,
                    depth_image=depth_image,
                    mask=mask,
                    intrinsics=intrinsics,
                    depth_scale=depth_scale,
                    detections=detections,
                )
                
                detection_info = f" ({len(detections)} detections)" if detections else " (no detections)"
                print(f"✓ Captured frame {frame_id}{detection_info}")
                last_capture_time = now

            # Handle other key presses
            if key == ord("d"):
                show_detections = not show_detections
                print(f"Detection overlay: {'ON' if show_detections else 'OFF'}")
            elif key == ord("q"):
                break

            frame_count += 1

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("RealSense pipeline stopped.")


if __name__ == "__main__":
    main()