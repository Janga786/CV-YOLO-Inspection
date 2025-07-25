from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm

DATASET_PATH = Path("/home/janga/YOLO_2/synthetic_dataset/images")

def convert_images_to_rgb():
    print(f"Converting images in: {DATASET_PATH} to RGB format...")
    converted_count = 0
    for image_path in tqdm(list(DATASET_PATH.glob("*.png")), desc="Converting images"):
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
                img.save(image_path) # Overwrite original
                converted_count += 1
            img.close()
        except Exception as e:
            print(f"Error converting {image_path}: {e}")
    
    print(f"\nFinished converting. {converted_count} images were converted to RGB.")

if __name__ == '__main__':
    convert_images_to_rgb()
