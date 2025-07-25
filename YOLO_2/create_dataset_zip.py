import zipfile
import os
from pathlib import Path

def create_dataset_zip():
    """Create a zip file of the synthetic dataset for Colab upload"""
    dataset_path = Path("/home/janga/YOLO_2/synthetic_dataset")
    zip_path = Path("/home/janga/YOLO_2/synthetic_dataset.zip")
    
    if not dataset_path.exists():
        print(f"Dataset directory {dataset_path} not found!")
        return
    
    print(f"Creating zip file: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Walk through the dataset directory
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                file_path = Path(root) / file
                # Get relative path for the zip file
                arcname = file_path.relative_to(dataset_path.parent)
                zipf.write(file_path, arcname)
                
    print(f"Dataset zip created successfully: {zip_path}")
    print(f"Zip file size: {zip_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Count files in the zip
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        file_count = len(zipf.namelist())
        print(f"Total files in zip: {file_count}")

if __name__ == "__main__":
    create_dataset_zip()