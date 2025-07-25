#!/usr/bin/env python3
"""
Resume script for synthetic data generation
Continues from where the previous generation left off
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_current_progress():
    """Check how many images are already generated"""
    current_dir = Path.cwd()
    images_dir = current_dir / "synthetic_dataset" / "images"
    labels_dir = current_dir / "synthetic_dataset" / "labels"
    
    if not images_dir.exists():
        print("❌ No synthetic_dataset/images directory found")
        return 0
    
    existing_images = list(images_dir.glob("*.png"))
    existing_labels = list(labels_dir.glob("*.txt"))
    
    # Count properly by checking the highest numbered image
    max_index = -1
    for img in existing_images:
        # Extract number from filename like "can_01348.png"
        try:
            name_parts = img.stem.split('_')
            if len(name_parts) >= 2:
                index = int(name_parts[-1])
                max_index = max(max_index, index)
        except ValueError:
            continue
    
    actual_count = max_index + 1 if max_index >= 0 else 0
    
    print(f"📊 Found {len(existing_images)} image files")
    print(f"📊 Highest index: {max_index}")
    print(f"📊 Next index to generate: {actual_count}")
    
    return actual_count

def run_resumed_generation():
    """Resume generation from where it left off"""
    current_dir = Path.cwd()
    output_dir = current_dir / "synthetic_dataset"
    
    # Check requirements first
    required_files = [
        "source/Coke Can.fbx",
        "textures/Label_Base_color.png",
        "hdri_backgrounds",
        "generate_detector_data.py"
    ]
    
    print("🔍 Checking requirements...")
    for req_file in required_files:
        if not (current_dir / req_file).exists():
            print(f"❌ Missing required file: {req_file}")
            return False
    
    # Check current progress
    current_count = check_current_progress()
    
    # Set target and batch size
    target_total = 3000  # Adjust this if you want a different target
    batch_size = 100
    
    if current_count >= target_total:
        print(f"✅ Already have {current_count} images. Target reached!")
        return True
    
    print(f"\n🎯 Target: {target_total} images")
    print(f"📦 Batch size: {batch_size} images")
    print(f"🔄 Resuming from image {current_count}")
    
    start_time = time.time()
    
    while current_count < target_total:
        remaining = target_total - current_count
        current_batch_size = min(batch_size, remaining)
        
        print(f"\n{'='*60}")
        print(f"🎨 Generating images {current_count} to {current_count + current_batch_size - 1}")
        print(f"📊 Progress: {current_count}/{target_total} ({(current_count/target_total)*100:.1f}%)")
        print(f"{'='*60}")
        
        # Build command
        cmd = [
            "blender",
            "--background",
            "--python", "generate_detector_data.py",
            "--",
            "--start_index", str(current_count),
            "--batch_size", str(current_batch_size),
            "--output_dir", str(output_dir),
            "--fbx_path", str(current_dir / "source" / "Coke Can.fbx"),
            "--texture_path", str(current_dir / "textures" / "Label_Base_color.png"),
            "--backgrounds_dir", str(current_dir / "hdri_backgrounds")
        ]
        
        # Run generation
        batch_start = time.time()
        try:
            print("🚀 Running Blender generation...")
            result = subprocess.run(cmd, check=False)
            
            if result.returncode != 0:
                print(f"⚠️  Batch returned code {result.returncode}")
            
            # Check what was actually generated
            new_count = check_current_progress()
            generated_this_batch = new_count - current_count
            
            if generated_this_batch > 0:
                batch_time = time.time() - batch_start
                per_image_time = batch_time / generated_this_batch
                print(f"✅ Generated {generated_this_batch} images in {batch_time:.1f}s ({per_image_time:.1f}s per image)")
                
                current_count = new_count
                
                # Progress update
                progress = (current_count / target_total) * 100
                elapsed = time.time() - start_time
                if current_count > 0:
                    eta = (elapsed / current_count) * (target_total - current_count)
                    eta_min = int(eta / 60)
                    eta_sec = int(eta % 60)
                    print(f"📈 Overall progress: {current_count}/{target_total} ({progress:.1f}%)")
                    print(f"⏱️  ETA: {eta_min}m {eta_sec}s")
            else:
                print("❌ No images generated in this batch")
                return False
                
        except KeyboardInterrupt:
            print("\n⛔ Generation interrupted by user")
            print(f"📊 Generated {current_count} images before interruption")
            return False
        except Exception as e:
            print(f"❌ Error in batch: {e}")
            return False
    
    # Final summary
    total_time = time.time() - start_time
    total_min = int(total_time / 60)
    total_sec = int(total_time % 60)
    
    print(f"\n{'='*60}")
    print(f"🎉 GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Total images: {current_count}")
    print(f"⏱️  Total time: {total_min}m {total_sec}s")
    print(f"📁 Images: {output_dir / 'images'}")
    print(f"📁 Labels: {output_dir / 'labels'}")
    
    return True

def main():
    """Main execution"""
    print("🔄 Resuming Synthetic Data Generation")
    print("="*60)
    
    if not run_resumed_generation():
        print("\n❌ Generation failed or was interrupted")
        return 1
    
    print("\n✅ Generation resumed successfully! 🎉")
    return 0

if __name__ == "__main__":
    sys.exit(main())