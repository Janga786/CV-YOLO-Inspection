from PIL import Image
import os
from pathlib import Path

DATASET_PATH = Path("/home/janga/YOLO_2/synthetic_dataset/images")

def check_images():
    print(f"Checking images in: {DATASET_PATH}")
    problematic_files = []
    for image_path in DATASET_PATH.glob("*.png"):
        try:
            img = Image.open(image_path)
            img.verify()  # Verify that it is an image
            img.close()
        except Exception as e:
            problematic_files.append(str(image_path))
            print(f"Error with {image_path}: {e}")
    
    if problematic_files:
        print("\nFound problematic image files:")
        for f in problematic_files:
            print(f)
        print("\nPlease remove or replace these files and try training again.")
    else:
        print("All PNG images seem to be valid.")

if __name__ == '__main__':
    check_images()
