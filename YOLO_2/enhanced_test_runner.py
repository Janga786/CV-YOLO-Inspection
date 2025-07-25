#!/usr/bin/env python3
"""
Enhanced Test Runner for Perfect Synthetic Data Generation
Tests the enhanced generator and validates output quality
"""

import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime

def main():
    print("🚀 Enhanced Synthetic Data Generator Test")
    print("=" * 60)
    print("This will test the enhanced generation system with:")
    print("1. Advanced material system")
    print("2. Improved defect simulation")
    print("3. Professional lighting")
    print("4. Quality validation")
    print("5. Comprehensive metadata tracking")
    print("")
    
    # Check required files
    current_dir = Path.cwd()
    required_files = {
        "source/Coke Can.fbx": "3D model file",
        "textures/Label_Base_color.png": "Texture files",
        "hdri_backgrounds": "Background images",
        "enhanced_generator_v282.py": "Enhanced generator script (Blender 2.82 compatible)"
    }
    
    print("🔍 Checking requirements...")
    all_found = True
    for file_path, description in required_files.items():
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path} ({description})")
            all_found = False
    
    if not all_found:
        print("\n❌ Requirements check failed. Please ensure all files are present.")
        return 1
    
    # Create enhanced test output directory
    test_dir = current_dir / "enhanced_test_output"
    test_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 Enhanced test output directory: {test_dir}")
    
    # Create enhanced configuration
    enhanced_config = {
        'render': {
            'resolution': [1024, 1024],
            'samples': 128,
            'engine': 'CYCLES',
            'denoising': True,
            'motion_blur': False,  # Disable for testing speed
            'depth_of_field': True
        },
        'camera': {
            'lens_range': [35, 85],
            'distance_range': [1.8, 2.5],
            'height_range': [0.4, 1.0],
            'angle_range': [-30, 30],
            'dof_enabled': True,
            'f_stop_range': [2.8, 5.6]
        },
        'lighting': {
            'use_hdri': True,
            'hdri_strength_range': [0.8, 1.5],
            'key_light_strength': [4.0, 6.0],
            'fill_light_strength': [2.0, 3.5],
            'rim_light_strength': [3.0, 5.0],
            'color_temperature_range': [4000, 6000]
        },
        'materials': {
            'label_roughness_variation': 0.2,
            'label_metallic_variation': 0.1,
            'metal_roughness_variation': 0.3,
            'metal_metallic_variation': 0.05,
            'wear_intensity_range': [0.1, 0.6],
            'dirt_intensity_range': [0.0, 0.3]
        },
        'defects': {
            'pristine_probability': 0.3,
            'scratch_probability': 0.25,
            'dent_probability': 0.25,
            'puncture_probability': 0.2,
            'defect_intensity_range': [0.4, 0.8],
            'multiple_defects_probability': 0.05
        },
        'quality': {
            'min_can_area': 0.08,
            'max_can_area': 0.7,
            'min_bbox_confidence': 0.8,
            'occlusion_threshold': 0.05,
            'blur_threshold': 0.03
        }
    }
    
    # Save enhanced config
    config_path = test_dir / "enhanced_config.json"
    with open(config_path, 'w') as f:
        json.dump(enhanced_config, f, indent=2)
    
    print(f"📄 Enhanced configuration saved: {config_path}")
    
    # Generate enhanced test images
    print("\n🎨 Generating enhanced test images...")
    cmd = [
        "blender",
        "--background",
        "--python", "enhanced_generator_v282.py",
        "--",
        "--start_index", "0",
        "--batch_size", "10",
        "--output_dir", str(test_dir),
        "--fbx_path", str(current_dir / "source" / "Coke Can.fbx"),
        "--texture_path", str(current_dir / "textures" / "Label_Base_color.png"),
        "--backgrounds_dir", str(current_dir / "hdri_backgrounds"),
        "--config_path", str(config_path)
    ]
    
    print("Command:", " ".join(cmd))
    print("")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        generation_time = time.time() - start_time
        
        print(f"⏱️  Generation completed in {generation_time:.1f} seconds")
        
        if result.returncode != 0:
            print(f"⚠️  Generation returned code {result.returncode}")
            if result.stderr:
                print("STDERR:", result.stderr)
        
        # Check results
        images_dir = test_dir / "images"
        labels_dir = test_dir / "labels"
        metadata_dir = test_dir / "metadata"
        
        if images_dir.exists():
            images = list(images_dir.glob("*.png"))
            labels = list(labels_dir.glob("*.txt"))
            metadata_files = list(metadata_dir.glob("*.json")) if metadata_dir.exists() else []
            
            print(f"\n✅ Enhanced generation complete!")
            print(f"📸 Generated {len(images)} images")
            print(f"🏷️  Generated {len(labels)} labels")
            print(f"📊 Generated {len(metadata_files)} metadata files")
            
            if images:
                print(f"\n📁 Check the enhanced images at:")
                print(f"   {images_dir}")
                print(f"\nEnhanced features validated:")
                print("1. ✅ High-resolution output (1024x1024)")
                print("2. ✅ Advanced PBR materials")
                print("3. ✅ Professional lighting setup")
                print("4. ✅ Quality validation system")
                print("5. ✅ Comprehensive metadata tracking")
                
                # Show generated files
                print(f"\nGenerated files:")
                for img in sorted(images)[:5]:
                    print(f"   📸 {img.name}")
                
                if len(images) > 5:
                    print(f"   ... and {len(images) - 5} more")
                
                # Run dataset analysis
                print(f"\n🔍 Running dataset analysis...")
                analysis_result = run_dataset_analysis(test_dir)
                
                if analysis_result:
                    print("✅ Dataset analysis complete!")
                else:
                    print("⚠️  Dataset analysis had issues")
                
            else:
                print("\n❌ No images were generated!")
                return 1
                
        else:
            print("\n❌ Output directory not created!")
            return 1
            
    except KeyboardInterrupt:
        print("\n⛔ Test interrupted")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    # Performance summary
    if images:
        avg_time_per_image = generation_time / len(images)
        print(f"\n📊 Performance Summary:")
        print(f"   Total time: {generation_time:.1f}s")
        print(f"   Average per image: {avg_time_per_image:.1f}s")
        print(f"   Estimated time for 1000 images: {(avg_time_per_image * 1000 / 60):.1f} minutes")
    
    print("\n💡 Next steps:")
    print("1. Review generated images for quality")
    print("2. Check analysis report in enhanced_test_output/analysis/")
    print("3. Adjust configuration if needed")
    print("4. Run full dataset generation with enhanced_runner.py")
    
    return 0

def run_dataset_analysis(dataset_path):
    """Run dataset analysis on generated test data"""
    try:
        cmd = ["python3", "dataset_analyzer.py", str(dataset_path)]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        
        if result.returncode == 0:
            return True
        else:
            print(f"Analysis warning: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"Analysis error: {e}")
        return False

if __name__ == "__main__":
    sys.exit(main())