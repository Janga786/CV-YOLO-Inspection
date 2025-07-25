#!/usr/bin/env python3
"""
Runner script for the professional Sketchfab Coca-Cola can model
Generates 3000 high-quality synthetic training images
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """Check all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check Blender
    try:
        result = subprocess.run(["blender", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ Blender found: {version_line}")
        else:
            print("❌ Blender not found or error running it")
            return False
    except FileNotFoundError:
        print("❌ Blender not installed or not in PATH")
        print("   Install with: sudo snap install blender --classic")
        return False
    
    # Check required files
    current_dir = Path.cwd()
    required_files = {
        "source/Coke Can.fbx": "Sketchfab model file",
        "textures/Label_Base_color.png": "Main texture file",
        "hdri_backgrounds": "Background images directory",
        "generate_detector_data.py": "Blender generation script"
    }
    
    all_found = True
    for file_path, description in required_files.items():
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path} ({description})")
            all_found = False
    
    if not all_found:
        return False
    
    # Check backgrounds
    bg_dir = current_dir / "hdri_backgrounds"
    bg_images = list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png"))
    if bg_images:
        print(f"✅ Found {len(bg_images)} background images")
    else:
        print("❌ No background images found in hdri_backgrounds/")
        return False
    
    # Check textures
    texture_dir = current_dir / "textures"
    texture_files = list(texture_dir.glob("*.png"))
    print(f"✅ Found {len(texture_files)} texture files")
    
    return True

def run_generation():
    """Run the dataset generation"""
    current_dir = Path.cwd()
    output_dir = current_dir / "dataset"
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Check existing progress
    existing_images = list(images_dir.glob("*.png"))
    existing_labels = list(labels_dir.glob("*.txt"))
    existing_count = min(len(existing_images), len(existing_labels))
    
    target_total = 3000
    batch_size = 100  # Generate 100 images at a time
    
    if existing_count >= target_total:
        print(f"✅ Already have {existing_count} images. Target reached!")
        return True
    
    if existing_count > 0:
        print(f"📊 Found {existing_count} existing images, resuming from there...")
    
    print(f"\n🎯 Target: {target_total} images")
    print(f"📦 Batch size: {batch_size} images")
    print(f"⚙️  Renderer: EEVEE (fast mode)")
    
    start_time = time.time()
    current_index = existing_count
    
    while current_index < target_total:
        remaining = target_total - current_index
        current_batch_size = min(batch_size, remaining)
        batch_num = (current_index // batch_size) + 1
        total_batches = (target_total + batch_size - 1) // batch_size
        
        print(f"\n{'='*60}")
        print(f"📊 Batch {batch_num}/{total_batches}")
        print(f"🎨 Generating images {current_index} to {current_index + current_batch_size - 1}")
        print(f"{'='*60}")
        
        # Build command
        cmd = [
            "blender",
            "--background",
            "--python", "generate_detector_data.py",
            "--",
            "--start_index", str(current_index),
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
                print(f"⚠️  Batch returned code {result.returncode}, checking output...")
            
            # Check what was actually generated
            new_images = list(images_dir.glob("*.png"))
            new_labels = list(labels_dir.glob("*.txt"))
            actual_count = min(len(new_images), len(new_labels))
            generated_this_batch = actual_count - existing_count
            
            if generated_this_batch > 0:
                batch_time = time.time() - batch_start
                per_image_time = batch_time / generated_this_batch
                print(f"✅ Generated {generated_this_batch} images in {batch_time:.1f}s ({per_image_time:.1f}s per image)")
                
                current_index += generated_this_batch
                existing_count = actual_count
                
                # Progress update
                progress = (existing_count / target_total) * 100
                elapsed = time.time() - start_time
                if existing_count > 0:
                    eta = (elapsed / existing_count) * (target_total - existing_count)
                    eta_min = int(eta / 60)
                    eta_sec = int(eta % 60)
                    print(f"📈 Overall progress: {existing_count}/{target_total} ({progress:.1f}%)")
                    print(f"⏱️  ETA: {eta_min}m {eta_sec}s")
            else:
                print("❌ No images generated in this batch")
                return False
                
        except KeyboardInterrupt:
            print("\n⛔ Generation interrupted by user")
            print(f"📊 Generated {existing_count} images before interruption")
            return False
        except Exception as e:
            print(f"❌ Error in batch: {e}")
            return False
    
    # Final summary
    total_time = time.time() - start_time
    total_min = int(total_time / 60)
    total_sec = int(total_time % 60)
    
    print(f"\n{'='*60}")
    print(f"🎉 DATASET GENERATION COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Total images: {existing_count}")
    print(f"⏱️  Total time: {total_min}m {total_sec}s")
    print(f"📁 Images: {images_dir}")
    print(f"📁 Labels: {labels_dir}")
    
    # Class distribution estimate
    print(f"\n📊 Estimated class distribution:")
    print(f"   - Class 0 (pristine): ~{int(existing_count * 0.4)} images (40%)")
    print(f"   - Class 1 (defective): ~{int(existing_count * 0.6)} images (60%)")
    print(f"     • Dents: ~{int(existing_count * 0.2)} images")
    print(f"     • Scratches: ~{int(existing_count * 0.2)} images")
    print(f"     • Punctures: ~{int(existing_count * 0.2)} images")
    
    return True

def organize_dataset():
    """Organize dataset into train/val splits"""
    print("\n📂 Organizing dataset for training...")
    
    dataset_dir = Path.cwd() / "dataset"
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    # Create YOLO structure
    yolo_dir = Path.cwd() / "yolo_dataset"
    train_images = yolo_dir / "train" / "images"
    train_labels = yolo_dir / "train" / "labels"
    val_images = yolo_dir / "val" / "images"
    val_labels = yolo_dir / "val" / "labels"
    
    # Create directories
    for dir_path in [train_images, train_labels, val_images, val_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = sorted(list(images_dir.glob("*.png")))
    
    if not image_files:
        print("❌ No images found to organize")
        return False
    
    # 80/20 train/val split
    num_images = len(image_files)
    num_val = int(num_images * 0.2)
    num_train = num_images - num_val
    
    print(f"📊 Splitting {num_images} images:")
    print(f"   - Training: {num_train} images (80%)")
    print(f"   - Validation: {num_val} images (20%)")
    
    # Shuffle for random split
    import random
    random.seed(42)  # For reproducibility
    shuffled_images = image_files.copy()
    random.shuffle(shuffled_images)
    
    # Split and copy files
    print("📁 Copying files...")
    
    for i, img_path in enumerate(shuffled_images):
        # Determine destination
        if i < num_train:
            dest_img_dir = train_images
            dest_lbl_dir = train_labels
            split = "train"
        else:
            dest_img_dir = val_images
            dest_lbl_dir = val_labels
            split = "val"
        
        # Copy image
        img_name = img_path.name
        lbl_name = img_path.stem + ".txt"
        lbl_path = labels_dir / lbl_name
        
        if lbl_path.exists():
            # Copy files
            import shutil
            shutil.copy2(img_path, dest_img_dir / img_name)
            shutil.copy2(lbl_path, dest_lbl_dir / lbl_name)
        
        # Progress
        if (i + 1) % 500 == 0:
            print(f"   Processed {i + 1}/{num_images} images...")
    
    # Create data.yaml for YOLO
    data_yaml = yolo_dir / "data.yaml"
    yaml_content = f"""# YOLOv8 Dataset Configuration
# Generated for Coca-Cola can defect detection

path: {yolo_dir.absolute()}
train: train/images
val: val/images

# Classes
nc: 2  # number of classes
names:
  0: pristine
  1: defective

# Class descriptions
# pristine: Can without any visible defects
# defective: Can with dents, scratches, or punctures
"""
    
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Dataset organized successfully!")
    print(f"📁 YOLO dataset: {yolo_dir}")
    print(f"📄 Config file: {data_yaml}")
    
    return True

def print_next_steps():
    """Print instructions for training"""
    print("\n" + "="*60)
    print("📋 NEXT STEPS - Training YOLOv8")
    print("="*60)
    
    print("\n1️⃣  Install YOLOv8 (if not already installed):")
    print("   pip install ultralytics")
    
    print("\n2️⃣  Train the model:")
    print("   python -c \"from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.train(data='yolo_dataset/data.yaml', epochs=100, imgsz=640)\"")
    
    print("\n3️⃣  For better results, use a larger model:")
    print("   python -c \"from ultralytics import YOLO; model = YOLO('yolov8s.pt'); model.train(data='yolo_dataset/data.yaml', epochs=100, imgsz=640, batch=16)\"")
    
    print("\n4️⃣  Monitor training progress:")
    print("   - Tensorboard logs will be in runs/detect/train")
    print("   - Best model will be saved as runs/detect/train/weights/best.pt")
    
    print("\n5️⃣  Test your model:")
    print("   python -c \"from ultralytics import YOLO; model = YOLO('runs/detect/train/weights/best.pt'); model.predict('path/to/test/image.jpg', show=True)\"")
    
    print("\n💡 Tips:")
    print("   - Adjust batch size based on your GPU memory")
    print("   - Use augmentation for better generalization")
    print("   - Consider training for more epochs (200-300) for best results")
    print("   - The synthetic data should give you a good starting point!")

def main():
    """Main execution"""
    print("🥤 Coca-Cola Can Defect Detection Dataset Generator")
    print("="*60)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues above.")
        return 1
    
    print("\n✅ All requirements satisfied!")
    
    # Run generation
    print("\n🚀 Starting dataset generation...")
    if not run_generation():
        print("\n❌ Dataset generation failed or was interrupted")
        return 1
    
    # Organize dataset
    if not organize_dataset():
        print("\n❌ Dataset organization failed")
        return 1
    
    # Print next steps
    print_next_steps()
    
    print("\n✅ All done! Happy training! 🎉")
    return 0

if __name__ == "__main__":
    sys.exit(main())
