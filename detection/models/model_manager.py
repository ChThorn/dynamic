#!/usr/bin/env python3
"""
Model management utility for downloading and organizing YOLO and SAM2 models.
Located within the detection module for proper package organization.
"""

import os
import requests
from pathlib import Path
from typing import Dict, List, Optional
import hashlib


class ModelManager:
    """Manages downloading and organizing detection models."""
    
    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize model manager with models directory."""
        if models_dir is None:
            # Default to models directory relative to this script (detection/models)
            models_dir = Path(__file__).parent
        
        self.models_dir = models_dir
        self.yolo_dir = models_dir / "yolo"
        self.sam2_dir = models_dir / "sam2"
        self.sam2_configs_dir = self.sam2_dir / "configs"
        self.sam2_checkpoints_dir = self.sam2_dir / "checkpoints"
        
        # Create directories
        self.yolo_dir.mkdir(parents=True, exist_ok=True)
        self.sam2_configs_dir.mkdir(parents=True, exist_ok=True)
        self.sam2_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, filepath: Path, description: str = "") -> bool:
        """Download file with progress indicator."""
        try:
            print(f"Downloading {description or filepath.name}...")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r  Progress: {progress:.1f}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)", end="")
            
            print(f"\n✓ Downloaded {filepath.name} successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {filepath.name}: {e}")
            if filepath.exists():
                filepath.unlink()  # Remove partial file
            return False
    
    def download_yolo_model(self, model_name: str) -> bool:
        """Download YOLO model by name (e.g., 'yolov8n', 'yolov8s')."""
        yolo_urls = {
            "yolov8n": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
            "yolov8s": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt", 
            "yolov8m": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
            "yolov8l": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt",
            "yolov8x": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
        }
        
        if model_name not in yolo_urls:
            print(f"✗ Unknown YOLO model: {model_name}")
            print(f"Available models: {list(yolo_urls.keys())}")
            return False
        
        filepath = self.yolo_dir / f"{model_name}.pt"
        if filepath.exists():
            print(f"✓ {model_name}.pt already exists")
            return True
        
        return self.download_file(yolo_urls[model_name], filepath, f"YOLO {model_name}")
    
    def download_sam2_config(self, config_name: str) -> bool:
        """Download SAM2 config file."""
        config_urls = {
            "sam2_hiera_tiny": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_t.yaml",
            "sam2_hiera_small": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_s.yaml", 
            "sam2_hiera_base_plus": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_b+.yaml",
            "sam2_hiera_large": "https://raw.githubusercontent.com/facebookresearch/sam2/main/sam2/configs/sam2/sam2_hiera_l.yaml",
        }
        
        if config_name not in config_urls:
            print(f"✗ Unknown SAM2 config: {config_name}")
            print(f"Available configs: {list(config_urls.keys())}")
            return False
        
        filepath = self.sam2_configs_dir / f"{config_name}.yaml"
        if filepath.exists():
            print(f"✓ {config_name}.yaml already exists")
            return True
        
        return self.download_file(config_urls[config_name], filepath, f"SAM2 config {config_name}")
    
    def download_sam2_checkpoint(self, model_name: str) -> bool:
        """Download SAM2 model checkpoint."""
        checkpoint_urls = {
            "sam2_hiera_tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
            "sam2_hiera_small": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
            "sam2_hiera_base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt", 
            "sam2_hiera_large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
        }
        
        if model_name not in checkpoint_urls:
            print(f"✗ Unknown SAM2 model: {model_name}")
            print(f"Available models: {list(checkpoint_urls.keys())}")
            return False
        
        filepath = self.sam2_checkpoints_dir / f"{model_name}.pth"
        if filepath.exists():
            print(f"✓ {model_name}.pth already exists")
            return True
        
        return self.download_file(checkpoint_urls[model_name], filepath, f"SAM2 checkpoint {model_name}")
    
    def download_sam2_model(self, model_name: str) -> bool:
        """Download both SAM2 config and checkpoint."""
        config_success = self.download_sam2_config(model_name)
        checkpoint_success = self.download_sam2_checkpoint(model_name)
        return config_success and checkpoint_success
    
    def setup_basic_models(self) -> bool:
        """Download basic models for development/testing."""
        print("Setting up basic models for development...")
        
        success = True
        
        # Download basic YOLO model
        if not self.download_yolo_model("yolov8n"):
            success = False
        
        # Download basic SAM2 model
        if not self.download_sam2_model("sam2_hiera_tiny"):
            success = False
        
        if success:
            print("\n✓ Basic models setup complete!")
            print("Models are now available in detection/models/")
            print("The capture script will automatically use these models.")
        else:
            print("\n✗ Some models failed to download. Check your internet connection.")
        
        return success
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models in the models directory."""
        available = {
            "yolo": [],
            "sam2_configs": [],
            "sam2_checkpoints": []
        }
        
        # Check YOLO models
        if self.yolo_dir.exists():
            available["yolo"] = [f.name for f in self.yolo_dir.glob("*.pt")]
        
        # Check SAM2 configs
        if self.sam2_configs_dir.exists():
            available["sam2_configs"] = [f.name for f in self.sam2_configs_dir.glob("*.yaml")]
        
        # Check SAM2 checkpoints  
        if self.sam2_checkpoints_dir.exists():
            available["sam2_checkpoints"] = [f.name for f in self.sam2_checkpoints_dir.glob("*.pth")]
        
        return available
    
    def print_model_status(self):
        """Print status of all models."""
        print("=== Detection Model Status ===")
        print(f"Models directory: {self.models_dir}")
        available = self.list_available_models()
        
        print("\n📁 YOLO Models:")
        if available["yolo"]:
            for model in available["yolo"]:
                size_mb = (self.yolo_dir / model).stat().st_size // 1024 // 1024
                print(f"  ✓ {model} ({size_mb}MB)")
        else:
            print("  (none downloaded)")
            print("  Run: python -m detection.models.model_manager --setup-basic")
        
        print("\n📁 SAM2 Configs:")
        if available["sam2_configs"]:
            for config in available["sam2_configs"]:
                print(f"  ✓ {config}")
        else:
            print("  (none downloaded)")
        
        print("\n📁 SAM2 Checkpoints:")
        if available["sam2_checkpoints"]:
            for checkpoint in available["sam2_checkpoints"]:
                size_mb = (self.sam2_checkpoints_dir / checkpoint).stat().st_size // 1024 // 1024
                print(f"  ✓ {checkpoint} ({size_mb}MB)")
        else:
            print("  (none downloaded)")
        
        # Check for complete SAM2 pairs
        configs = {f.stem for f in self.sam2_configs_dir.glob("*.yaml")}
        checkpoints = {f.stem for f in self.sam2_checkpoints_dir.glob("*.pth")}
        complete_pairs = configs.intersection(checkpoints)
        
        print(f"\n🔗 Complete SAM2 pairs: {len(complete_pairs)}")
        for pair in complete_pairs:
            print(f"  ✓ {pair}")
        
        # Show what the capture script will use
        print(f"\n🎯 Current capture script configuration:")
        print(f"  YOLO: detection/models/yolo/yolov8n.pt {'✓' if (self.yolo_dir / 'yolov8n.pt').exists() else '✗'}")
        print(f"  SAM2: sam2_hiera_tiny {'✓' if 'sam2_hiera_tiny' in complete_pairs else '✗'}")


def main():
    """Command line interface for model management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download and manage detection models")
    parser.add_argument("--setup-basic", action="store_true", help="Download basic models for development")
    parser.add_argument("--yolo", help="Download specific YOLO model (e.g., yolov8n, yolov8s)")
    parser.add_argument("--sam2", help="Download specific SAM2 model (e.g., sam2_hiera_tiny)")
    parser.add_argument("--status", action="store_true", help="Show status of all models")
    parser.add_argument("--models-dir", type=Path, help="Custom models directory")
    
    args = parser.parse_args()
    
    manager = ModelManager(args.models_dir)
    
    if args.status:
        manager.print_model_status()
    elif args.setup_basic:
        manager.setup_basic_models()
    elif args.yolo:
        manager.download_yolo_model(args.yolo)
    elif args.sam2:
        manager.download_sam2_model(args.sam2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()