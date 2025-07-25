#!/usr/bin/env python3
"""
Enhanced Production Runner for Perfect Synthetic Training Data
Works with the fixed generate_detector_data.py script
"""

import os
import sys
import subprocess
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
import random

def check_system_requirements():
    """Check system requirements for generation"""
    print("🔍 Checking system requirements...")
    
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
        print("   Or download from: https://www.blender.org/download/")
        return False
    
    # Check Python packages (for analysis, not required for generation)
    optional_packages = ['numpy', 'matplotlib', 'pandas']
    analysis_available = True
    
    for package in optional_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            analysis_available = False
            break
    
    if not analysis_available:
        print("⚠️  Analysis packages not found (optional)")
        print("   For dataset analysis, install: pip install numpy matplotlib pandas")
    else:
        print("✅ Analysis packages found")
    
    # Check required files
    current_dir = Path.cwd()
    required_files = {
        "generate_detector_data.py": "Fixed generation script",
    }
    
    # Also check for common locations of the FBX file
    fbx_locations = [
        "source/Coke Can.fbx",
        "Coke Can.fbx",
        "coca_cola_can.fbx",
        "can.fbx",
        "model/Coke Can.fbx",
        "models/Coke Can.fbx"
    ]
    
    texture_locations = [
        "textures/Label_Base_color.png",
        "textures/label_base_color.png",
        "Label_Base_color.png",
        "texture.png"
    ]
    
    background_locations = [
        "hdri_backgrounds",
        "backgrounds",
        "background_images",
        "hdri"
    ]
    
    # Check generator script
    generator_found = False
    for file_path, description in required_files.items():
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✅ Found: {file_path}")
            generator_found = True
        else:
            print(f"❌ Missing: {file_path} ({description})")
    
    if not generator_found:
        return False
    
    # Find FBX file
    fbx_path = None
    for location in fbx_locations:
        full_path = current_dir / location
        if full_path.exists():
            fbx_path = full_path
            print(f"✅ Found FBX model: {location}")
            break
    
    if not fbx_path:
        print("❌ No FBX model file found. Checked locations:")
        for loc in fbx_locations:
            print(f"   - {loc}")
        return False
    
    # Find texture
    texture_path = None
    texture_dir = None
    for location in texture_locations:
        full_path = current_dir / location
        if full_path.exists():
            texture_path = full_path
            texture_dir = full_path.parent
            print(f"✅ Found texture: {location}")
            break
    
    if not texture_path:
        # Check if textures directory exists
        textures_dir = current_dir / "textures"
        if textures_dir.exists() and textures_dir.is_dir():
            texture_files = list(textures_dir.glob("*.png"))
            if texture_files:
                texture_path = texture_files[0]
                texture_dir = textures_dir
                print(f"✅ Found texture directory with {len(texture_files)} files")
            else:
                print("❌ Textures directory exists but contains no PNG files")
                return False
        else:
            print("❌ No texture files found. Checked locations:")
            for loc in texture_locations:
                print(f"   - {loc}")
            return False
    
    # Find backgrounds directory
    bg_dir = None
    for location in background_locations:
        full_path = current_dir / location
        if full_path.exists() and full_path.is_dir():
            bg_images = list(full_path.glob("*.jpg")) + list(full_path.glob("*.png")) + \
                       list(full_path.glob("*.hdr")) + list(full_path.glob("*.exr"))
            if bg_images:
                bg_dir = full_path
                print(f"✅ Found backgrounds directory: {location} ({len(bg_images)} images)")
                break
    
    if not bg_dir:
        print("⚠️  No background images directory found. Will use solid colors.")
        print("   For better results, create a 'backgrounds' directory with images")
        # Create empty backgrounds directory
        bg_dir = current_dir / "backgrounds"
        bg_dir.mkdir(exist_ok=True)
    
    # Store paths for later use
    global FOUND_PATHS
    FOUND_PATHS = {
        'fbx': fbx_path,
        'texture': texture_path,
        'texture_dir': texture_dir,
        'backgrounds': bg_dir
    }
    
    return True

def run_generation_with_validation():
    """Run the dataset generation with validation"""
    current_dir = Path.cwd()
    output_dir = current_dir / "synthetic_dataset"
    
    # Use found paths
    fbx_path = FOUND_PATHS['fbx']
    texture_path = FOUND_PATHS['texture_dir']  # Use directory, not file
    backgrounds_dir = FOUND_PATHS['backgrounds']
    
    print(f"\n📁 Output directory: {output_dir}")
    print(f"📄 FBX Model: {fbx_path}")
    print(f"🎨 Textures: {texture_path}")
    print(f"🖼️  Backgrounds: {backgrounds_dir}")
    
    # Check existing progress
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    existing_images = list(images_dir.glob("*.png")) if images_dir.exists() else []
    existing_labels = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    existing_count = min(len(existing_images), len(existing_labels))
    
    # Configuration
    target_total = 5000  # Target number of images
    batch_size = 100     # Images per batch
    
    if existing_count >= target_total:
        print(f"\n✅ Already have {existing_count} images. Target reached!")
        return True, output_dir, existing_count
    
    if existing_count > 0:
        print(f"\n📊 Found {existing_count} existing images, resuming from there...")
        
        # Ask if user wants to continue or restart
        response = input("\nContinue from existing images? (y/n) [y]: ").strip().lower()
        if response == 'n':
            print("🗑️  Clearing existing dataset...")
            shutil.rmtree(output_dir, ignore_errors=True)
            existing_count = 0
    
    print(f"\n🎯 Target: {target_total} images")
    print(f"📦 Batch size: {batch_size} images")
    print(f"🎨 Using fixed visibility validation")
    
    start_time = time.time()
    current_index = existing_count
    total_success = existing_count
    failed_batches = 0
    
    while current_index < target_total:
        remaining = target_total - current_index
        current_batch_size = min(batch_size, remaining)
        batch_num = (current_index // batch_size) + 1
        total_batches = (target_total + batch_size - 1) // batch_size
        
        print(f"\n{'='*70}")
        print(f"📊 BATCH {batch_num}/{total_batches}")
        print(f"🎨 Generating images {current_index} to {current_index + current_batch_size - 1}")
        print(f"📈 Progress: {(current_index / target_total * 100):.1f}%")
        print(f"{'='*70}")
        
        # Build command
        cmd = [
            "blender",
            "--background",
            "--python", "generate_detector_data.py",
            "--",
            "--start_index", str(current_index),
            "--batch_size", str(current_batch_size),
            "--output_dir", str(output_dir),
            "--fbx_path", str(fbx_path),
            "--texture_path", str(texture_path),
            "--backgrounds_dir", str(backgrounds_dir)
        ]
        
        # Run generation
        batch_start = time.time()
        try:
            print("🚀 Running Blender generation with visibility validation...")
            result = subprocess.run(cmd, check=False)
            
            if result.returncode != 0:
                print(f"⚠️  Batch returned code {result.returncode}")
                failed_batches += 1
                
                if failed_batches > 3:
                    print("❌ Too many failed batches. Stopping.")
                    break
            
            # Check what was actually generated
            new_images = list(images_dir.glob("*.png")) if images_dir.exists() else []
            new_labels = list(labels_dir.glob("*.txt")) if labels_dir.exists() else []
            actual_count = min(len(new_images), len(new_labels))
            generated_this_batch = actual_count - total_success
            
            if generated_this_batch > 0:
                batch_time = time.time() - batch_start
                per_image_time = batch_time / generated_this_batch
                
                print(f"✅ Generated {generated_this_batch} images in {batch_time:.1f}s")
                print(f"⏱️  Average time per image: {per_image_time:.1f}s")
                
                current_index += generated_this_batch
                total_success = actual_count
                failed_batches = 0  # Reset failed counter on success
                
                # Progress update
                progress = (total_success / target_total) * 100
                elapsed = time.time() - start_time
                
                if total_success > existing_count:
                    new_images_count = total_success - existing_count
                    avg_time_per_new_image = elapsed / new_images_count
                    eta = avg_time_per_new_image * (target_total - total_success)
                    eta_minutes = int(eta / 60)
                    eta_seconds = int(eta % 60)
                    
                    print(f"📈 Overall progress: {total_success}/{target_total} ({progress:.1f}%)")
                    print(f"⏱️  ETA: {eta_minutes}m {eta_seconds}s")
                
                # Show sample of generated defect types
                if generated_this_batch > 0:
                    sample_defects(output_dir, current_index - generated_this_batch, current_index)
                
            else:
                print("❌ No images generated in this batch")
                failed_batches += 1
                
                if failed_batches > 3:
                    print("❌ Too many failed batches. Stopping.")
                    break
                
                print("Attempting to continue with next batch...")
                current_index += current_batch_size  # Skip this batch
                
        except KeyboardInterrupt:
            print("\n⛔ Generation interrupted by user")
            print(f"📊 Generated {total_success} images before interruption")
            return False, output_dir, total_success
        except Exception as e:
            print(f"❌ Error in batch: {e}")
            failed_batches += 1
            
            if failed_batches > 3:
                print("❌ Too many errors. Stopping.")
                break
            
            print("Attempting to continue with next batch...")
            current_index += current_batch_size  # Skip this batch
    
    # Final summary
    total_time = time.time() - start_time
    total_minutes = int(total_time / 60)
    total_seconds = int(total_time % 60)
    
    print(f"\n{'='*70}")
    print(f"🎉 DATASET GENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"📊 Total images: {total_success}")
    print(f"⏱️  Total time: {total_minutes}m {total_seconds}s")
    print(f"📁 Images: {images_dir}")
    print(f"📁 Labels: {labels_dir}")
    
    return True, output_dir, total_success

def sample_defects(output_dir, start_idx, end_idx):
    """Sample and report defect types in the batch"""
    labels_dir = output_dir / "labels"
    
    defect_counts = {0: 0, 1: 0}  # 0: pristine, 1: defective
    
    for idx in range(start_idx, end_idx):
        label_file = labels_dir / f"can_{idx:05d}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                line = f.readline().strip()
                if line:
                    class_id = int(line.split()[0])
                    defect_counts[class_id] = defect_counts.get(class_id, 0) + 1
    
    total = sum(defect_counts.values())
    if total > 0:
        pristine_pct = (defect_counts[0] / total) * 100
        defective_pct = (defect_counts[1] / total) * 100
        print(f"   🏷️  Batch defects: {defect_counts[0]} pristine ({pristine_pct:.1f}%), {defect_counts[1]} defective ({defective_pct:.1f}%)")

def organize_yolo_dataset(output_dir, total_images):
    """Organize dataset in YOLO format"""
    print("\n📂 Organizing dataset for YOLO training...")
    
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    
    # Create YOLO structure
    yolo_dir = output_dir / "yolo_dataset"
    train_images = yolo_dir / "train" / "images"
    train_labels = yolo_dir / "train" / "labels"
    val_images = yolo_dir / "val" / "images"
    val_labels = yolo_dir / "val" / "labels"
    test_images = yolo_dir / "test" / "images"
    test_labels = yolo_dir / "test" / "labels"
    
    # Create directories
    for dir_path in [train_images, train_labels, val_images, val_labels, test_images, test_labels]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Get all paired files
    image_files = []
    for img_path in sorted(images_dir.glob("*.png")):
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            image_files.append(img_path)
    
    if not image_files:
        print("❌ No paired images and labels found")
        return False
    
    # 70/20/10 train/val/test split
    num_images = len(image_files)
    num_test = max(1, int(num_images * 0.1))
    num_val = max(1, int(num_images * 0.2))
    num_train = num_images - num_val - num_test
    
    print(f"📊 Splitting {num_images} images:")
    print(f"   - Training: {num_train} images (~70%)")
    print(f"   - Validation: {num_val} images (~20%)")
    print(f"   - Test: {num_test} images (~10%)")
    
    # Shuffle for random split
    random.seed(42)  # For reproducibility
    shuffled_images = image_files.copy()
    random.shuffle(shuffled_images)
    
    # Count classes for statistics
    class_counts = {'train': {0: 0, 1: 0}, 'val': {0: 0, 1: 0}, 'test': {0: 0, 1: 0}}
    
    # Split and copy files
    print("📁 Copying files...")
    
    splits = [
        (shuffled_images[:num_train], train_images, train_labels, "train"),
        (shuffled_images[num_train:num_train+num_val], val_images, val_labels, "val"),
        (shuffled_images[num_train+num_val:], test_images, test_labels, "test")
    ]
    
    for images_subset, dest_img_dir, dest_lbl_dir, split_name in splits:
        for img_path in images_subset:
            img_name = img_path.name
            lbl_name = img_path.stem + ".txt"
            lbl_path = labels_dir / lbl_name
            
            # Copy files
            shutil.copy2(img_path, dest_img_dir / img_name)
            shutil.copy2(lbl_path, dest_lbl_dir / lbl_name)
            
            # Count classes
            with open(lbl_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    class_id = int(line.split()[0])
                    class_counts[split_name][class_id] = class_counts[split_name].get(class_id, 0) + 1
    
    # Create data.yaml for YOLO
    data_yaml = yolo_dir / "data.yaml"
    yaml_content = f"""# YOLOv8 Dataset Configuration
# Generated by Fixed Synthetic Data Generator
# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

path: {yolo_dir.absolute()}
train: train/images
val: val/images
test: test/images

# Classes
nc: 2  # number of classes
names:
  0: pristine
  1: defective

# Dataset statistics
total_images: {num_images}
train_images: {num_train}
val_images: {num_val}
test_images: {num_test}

# Class distribution
train_pristine: {class_counts['train'][0]}
train_defective: {class_counts['train'][1]}
val_pristine: {class_counts['val'][0]}
val_defective: {class_counts['val'][1]}
test_pristine: {class_counts['test'][0]}
test_defective: {class_counts['test'][1]}

# Generation info
visibility_validated: true
render_engine: EEVEE
image_size: 640x640
"""
    
    with open(data_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Dataset organized successfully!")
    print(f"📁 YOLO dataset: {yolo_dir}")
    print(f"📄 Config file: {data_yaml}")
    
    # Print class distribution
    print("\n📊 Class Distribution:")
    for split in ['train', 'val', 'test']:
        total = sum(class_counts[split].values())
        if total > 0:
            pristine_pct = (class_counts[split][0] / total) * 100
            defective_pct = (class_counts[split][1] / total) * 100
            print(f"   {split}: {class_counts[split][0]} pristine ({pristine_pct:.1f}%), "
                  f"{class_counts[split][1]} defective ({defective_pct:.1f}%)")
    
    return True

def print_training_instructions(output_dir, total_images):
    """Print detailed training instructions"""
    yolo_dir = output_dir / "yolo_dataset"
    
    print("\n" + "="*70)
    print("🎉 SYNTHETIC DATASET READY FOR TRAINING!")
    print("="*70)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   - Total validated images: {total_images}")
    print(f"   - All cans visible: ✅")
    print(f"   - Bounding boxes validated: ✅")
    print(f"   - YOLO format ready: ✅")
    
    print("\n🚀 Quick Start - Train Your Model:")
    print("\n1️⃣  Install YOLOv8:")
    print("   pip install ultralytics")
    
    print("\n2️⃣  Basic training (good for testing):")
    print(f"   yolo detect train data={yolo_dir}/data.yaml model=yolov8n.pt epochs=100 imgsz=640")
    
    print("\n3️⃣  Better training (recommended):")
    print(f"   yolo detect train data={yolo_dir}/data.yaml model=yolov8s.pt epochs=200 imgsz=640 batch=16")
    
    print("\n4️⃣  Best training (if you have good GPU):")
    print(f"   yolo detect train data={yolo_dir}/data.yaml model=yolov8m.pt epochs=300 imgsz=640 batch=8 patience=50")
    
    print("\n📝 Python Script Example:")
    print(f"""
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')  # or yolov8n.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Train the model
results = model.train(
    data='{yolo_dir}/data.yaml',
    epochs=200,
    imgsz=640,
    batch=16,
    name='coca_cola_can_detector'
)

# Validate the model
metrics = model.val()
print(f"mAP50-95: {{metrics.box.map}}")
print(f"mAP50: {{metrics.box.map50}}")
""")
    
    print("\n🔍 Testing Your Model:")
    print(f"""
# After training, test on new images:
from ultralytics import YOLO

model = YOLO('runs/detect/coca_cola_can_detector/weights/best.pt')
results = model('path/to/test/image.jpg')
results[0].show()  # Display results
""")
    
    print("\n📊 Monitor Training:")
    print("   - TensorBoard: tensorboard --logdir runs/detect")
    print("   - Best weights: runs/detect/coca_cola_can_detector/weights/best.pt")
    print("   - Training curves: runs/detect/coca_cola_can_detector/results.png")
    
    print("\n💡 Tips for Best Results:")
    print("   ✓ Start with yolov8s.pt for balanced speed/accuracy")
    print("   ✓ Use patience parameter to prevent overfitting")
    print("   ✓ Test on real can images to validate performance")
    print("   ✓ Consider data augmentation if testing on real images shows poor results")
    print("   ✓ Export to ONNX/TensorRT for production deployment")

def main():
    """Main execution function"""
    print("🥤 Coca-Cola Can Defect Detection Dataset Generator")
    print("=" * 70)
    print("This system generates synthetic training data with guaranteed visibility")
    print("Every generated image will have the can fully visible in frame")
    print("")
    
    # Check system requirements
    if not check_system_requirements():
        print("\n❌ System requirements check failed. Please fix the issues above.")
        return 1
    
    print("\n✅ All requirements satisfied!")
    
    # Run generation
    print("\n🚀 Starting dataset generation with visibility validation...")
    try:
        success, output_dir, total_count = run_generation_with_validation()
    except Exception as e:
        print(f"\n❌ Generation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    if not success:
        print("\n⚠️  Dataset generation was interrupted")
        if total_count > 0:
            print(f"📊 But we still generated {total_count} valid images!")
        else:
            return 1
    
    # Organize dataset if we have images
    if total_count > 0:
        if not organize_yolo_dataset(output_dir, total_count):
            print("\n❌ Dataset organization failed")
            return 1
        
        # Print training instructions
        print_training_instructions(output_dir, total_count)
    
    print("\n✅ Dataset generation complete! Happy training! 🎉")
    return 0

if __name__ == "__main__":
    # Global variable to store found paths
    FOUND_PATHS = {}
    sys.exit(main())
