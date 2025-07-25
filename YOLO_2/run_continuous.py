#!/usr/bin/env python3
"""
Continuous generation script that runs without timeout issues
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def get_current_count():
    """Get current image count"""
    images_dir = Path("synthetic_dataset/images")
    if not images_dir.exists():
        return 0
    
    existing_images = list(images_dir.glob("*.png"))
    max_index = -1
    for img in existing_images:
        try:
            name_parts = img.stem.split('_')
            if len(name_parts) >= 2:
                index = int(name_parts[-1])
                max_index = max(max_index, index)
        except ValueError:
            continue
    
    return max_index + 1 if max_index >= 0 else 0

def run_batch(start_index, batch_size=50):
    """Run a single batch"""
    cmd = [
        "blender",
        "--background",
        "--python", "generate_detector_data.py",
        "--",
        "--start_index", str(start_index),
        "--batch_size", str(batch_size),
        "--output_dir", "synthetic_dataset",
        "--fbx_path", "source/Coke Can.fbx",
        "--texture_path", "textures/Label_Base_color.png",
        "--backgrounds_dir", "hdri_backgrounds"
    ]
    
    print(f"🎨 Generating batch {start_index}-{start_index + batch_size - 1}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 minute timeout
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⚠️  Batch timed out")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    target = 3000
    batch_size = 50
    
    while True:
        current_count = get_current_count()
        
        if current_count >= target:
            print(f"🎉 Complete! Generated {current_count} images")
            break
        
        print(f"📊 Progress: {current_count}/{target} ({(current_count/target)*100:.1f}%)")
        
        if run_batch(current_count, batch_size):
            print("✅ Batch completed successfully")
        else:
            print("❌ Batch failed, retrying...")
            time.sleep(5)

if __name__ == "__main__":
    main()