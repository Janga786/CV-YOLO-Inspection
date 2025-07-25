#!/usr/bin/env python3
"""
Quick test script to verify the can generation is working correctly
Generates just 5 test images to check positioning and textures
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("🧪 Coca-Cola Can Generation Test")
    print("=" * 50)
    print("This will generate 5 test images to verify:")
    print("1. Can is properly positioned in frame")
    print("2. All textures are applied correctly")
    print("3. Defects are visible")
    print("")
    
    # Check required files
    current_dir = Path.cwd()
    fbx_path = current_dir / "source" / "Coke Can.fbx"
    
    if not fbx_path.exists():
        print(f"❌ Model not found: {fbx_path}")
        return 1
    
    # Create test output directory
    test_dir = current_dir / "test_output"
    test_dir.mkdir(exist_ok=True)
    
    print(f"📁 Test output directory: {test_dir}")
    print("")
    
    # Generate 5 test images with different defect types
    cmd = [
        "blender",
        "--background",
        "--python", "generate_detector_data.py",
        "--",
        "--start_index", "0",
        "--batch_size", "5",
        "--output_dir", str(test_dir),
        "--fbx_path", str(fbx_path),
        "--texture_path", str(current_dir / "textures" / "Label_Base_color.png"),
        "--backgrounds_dir", str(current_dir / "hdri_backgrounds")
    ]
    
    print("🚀 Running test generation...")
    print("Command:", " ".join(cmd))
    print("")
    
    try:
        result = subprocess.run(cmd, check=False)
        
        # Check results
        images_dir = test_dir / "images"
        labels_dir = test_dir / "labels"
        
        if images_dir.exists():
            images = list(images_dir.glob("*.png"))
            labels = list(labels_dir.glob("*.txt"))
            
            print(f"\n✅ Test complete!")
            print(f"📸 Generated {len(images)} images")
            print(f"🏷️  Generated {len(labels)} labels")
            
            if images:
                print(f"\n📁 Check the test images at:")
                print(f"   {images_dir}")
                print(f"\nVerify that:")
                print("1. The Coca-Cola can is visible in each image")
                print("2. The red Coca-Cola label texture is applied")
                print("3. The metallic can body texture is visible")
                print("4. The can is properly centered and sized")
                
                # Show first few image names
                print(f"\nGenerated files:")
                for img in sorted(images)[:5]:
                    print(f"   - {img.name}")
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
    
    print("\n💡 If the test images look good, run the full generation with:")
    print("   python3 sketchfab_runner.py")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
