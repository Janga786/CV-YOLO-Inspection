#!/usr/bin/env python3
"""
Download damaged coke cans and defect overlay images
Uses multiple sources to get realistic defect images
"""

import requests
import os
import time
from pathlib import Path
from tqdm import tqdm
import json

class DamagedCanDownloader:
    def __init__(self, output_dir="damaged_cans_dataset"):
        self.output_dir = Path(output_dir)
        
        # Create directory structure
        self.damaged_cans_dir = self.output_dir / "damaged_cans"
        self.defect_overlays_dir = self.output_dir / "defect_overlays"
        self.scratch_overlays_dir = self.defect_overlays_dir / "scratches"
        self.dent_overlays_dir = self.defect_overlays_dir / "dents"
        self.hole_overlays_dir = self.defect_overlays_dir / "holes"
        
        # Create all directories
        for dir_path in [self.damaged_cans_dir, self.scratch_overlays_dir, 
                         self.dent_overlays_dir, self.hole_overlays_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # API keys (you'll need to add your own)
        self.pexels_api_key = "NZVDUv6JNrJynNuyh2DhJAL14sEnx22ok29RnbiHzFz8qZeW9hUHHX22"  # Get from https://www.pexels.com/api/
        
    def download_file(self, url, filepath, description=""):
        """Download a file with progress bar"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, 
                         desc=description) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False
    
    def search_pexels(self, query, per_page=30):
        """Search Pexels for images"""
        if self.pexels_api_key == "YOUR_PEXELS_API_KEY":
            print("Please set your Pexels API key!")
            return []
        
        headers = {"Authorization": self.pexels_api_key}
        params = {"query": query, "per_page": per_page}
        
        try:
            response = requests.get("https://api.pexels.com/v1/search", 
                                  headers=headers, params=params)
            response.raise_for_status()
            return response.json().get("photos", [])
        except Exception as e:
            print(f"Pexels search error: {e}")
            return []
    
    def download_damaged_cans(self):
        """Download images of damaged/dented coke cans"""
        print("\n=== Downloading Damaged Coke Cans ===")
        
        search_queries = [
            "damaged coke can",
            "dented coca cola can",
            "crushed coke can",
            "scratched aluminum can",
            "punctured soda can",
            "damaged aluminum beverage can",
            "recycled crushed cans",
            "dented soda cans"
        ]
        
        downloaded = 0
        
        # Search Pexels
        print("\nSearching Pexels...")
        for query in search_queries:
            print(f"  Searching: {query}")
            photos = self.search_pexels(query, per_page=20)
            
            for photo in photos:
                filename = f"pexels_damaged_{photo['id']}.jpg"
                filepath = self.damaged_cans_dir / filename
                
                if not filepath.exists():
                    url = photo['src']['large']
                    if self.download_file(url, filepath, f"Damaged can {downloaded+1}"):
                        downloaded += 1
            
            time.sleep(1)  # Rate limiting
        
        print(f"\nDownloaded {downloaded} damaged can images")
    
    def download_defect_overlays(self):
        """Download transparent defect overlays"""
        print("\n=== Downloading Defect Overlays ===")
        
        # For overlays, we need transparent PNGs
        # These searches will help find textures that can be used as overlays
        
        overlay_searches = {
            "scratches": [
                "metal scratch texture transparent png",
                "scratch overlay png",
                "scratch marks transparent",
                "metal scratches alpha channel",
                "brush scratches png"
            ],
            "dents": [
                "dent texture transparent",
                "impact damage overlay",
                "metal dent texture png",
                "bump map dent"
            ],
            "holes": [
                "bullet hole transparent png",
                "puncture hole overlay",
                "hole texture alpha",
                "torn metal png"
            ]
        }
        
        # Note: Finding true transparent overlays via API is challenging
        # You might need to manually source these or create them
        print("\nNote: For best results with overlays, consider:")
        print("1. Searching on PNG repositories like PNGTree, FreePNG")
        print("2. Creating your own using photo editing software")
        print("3. Using texture generation tools")
        
        # Download some texture examples that could be processed
        print("\nDownloading texture examples...")
        
        texture_urls = {
            "scratches": [
                # Add direct URLs to scratch textures if you find them
            ],
            "dents": [
                # Add direct URLs to dent textures if you find them
            ],
            "holes": [
                # Add direct URLs to hole textures if you find them
            ]
        }
        
        # Create a manifest file
        manifest = {
            "damaged_cans": str(self.damaged_cans_dir),
            "defect_overlays": {
                "scratches": str(self.scratch_overlays_dir),
                "dents": str(self.dent_overlays_dir),
                "holes": str(self.hole_overlays_dir)
            },
            "notes": "Add transparent PNG overlays to the defect directories"
        }
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"\nCreated manifest at: {manifest_path}")
    
    def download_from_urls(self, urls_file):
        """Download from a list of URLs"""
        urls_path = Path(urls_file)
        if not urls_path.exists():
            print(f"URLs file not found: {urls_file}")
            return
        
        with open(urls_path, 'r') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        print(f"\nDownloading {len(urls)} files from URLs list...")
        
        for i, url in enumerate(urls):
            filename = f"custom_{i:04d}_{Path(url).name}"
            filepath = self.damaged_cans_dir / filename
            
            if not filepath.exists():
                self.download_file(url, filepath, f"File {i+1}/{len(urls)}")
    
    def create_overlay_instructions(self):
        """Create instructions for getting overlay images"""
        instructions = """
# Defect Overlay Instructions

To create realistic defect overlays, you need transparent PNG images. Here are the best sources:

## 1. Free PNG Websites (Manual Download)
- **PNGTree**: https://pngtree.com/ (search for "scratch png", "metal damage png")
- **FreePNG**: https://freepng.com/ (search for "scratch overlay", "hole png")
- **PNGWing**: https://www.pngwing.com/ (search for "damage texture")
- **CleanPNG**: https://www.cleanpng.com/ (various damage effects)

## 2. Create Your Own Overlays
Using GIMP or Photoshop:
1. Find high-contrast damage photos
2. Convert to grayscale
3. Adjust levels to make damage pure black/white
4. Delete white background (make transparent)
5. Save as PNG with alpha channel

## 3. Texture Resources
- **Textures.com**: https://www.textures.com/ (formerly CGTextures)
- **Poliigon**: https://www.poliigon.com/ (professional textures)
- **TextureHaven**: https://texturehaven.com/ (free PBR textures)

## 4. Search Terms for Best Results
### Scratches:
- "metal scratch alpha map"
- "scratch brush png transparent"
- "grunge scratch overlay"

### Dents:
- "impact damage normal map"
- "dent displacement map"
- "metal deformation texture"

### Holes/Punctures:
- "bullet hole transparent png"
- "torn metal alpha"
- "puncture wound overlay"

## 5. Processing Tips
- Overlays should be mostly transparent with defects in black/dark gray
- Higher resolution (2K-4K) works better for detail
- Multiple variations of each defect type improve realism
- Consider rotation and scaling when applying

## Directory Structure
Place downloaded overlays in:
- {scratch_dir}
- {dent_dir}  
- {hole_dir}
""".format(
            scratch_dir=self.scratch_overlays_dir,
            dent_dir=self.dent_overlays_dir,
            hole_dir=self.hole_overlays_dir
        )
        
        instructions_path = self.output_dir / "overlay_instructions.txt"
        with open(instructions_path, 'w') as f:
            f.write(instructions)
        
        print(f"\nCreated overlay instructions at: {instructions_path}")
    
    def run(self):
        """Run the complete download process"""
        print("=== Damaged Coke Can Dataset Downloader ===")
        print(f"Output directory: {self.output_dir}")
        
        # Download damaged cans
        self.download_damaged_cans()
        
        # Set up overlay directories and instructions
        self.download_defect_overlays()
        self.create_overlay_instructions()
        
        print("\n=== Download Summary ===")
        print(f"Damaged cans: {len(list(self.damaged_cans_dir.glob('*')))} files")
        print(f"Overlay directories created at: {self.defect_overlays_dir}")
        print("\nNext steps:")
        print("1. Add your API keys to the script")
        print("2. Download transparent overlay PNGs manually")
        print("3. Use the overlay compositor script to create synthetic damaged cans")


def main():
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Download damaged can images and overlays")
    parser.add_argument("--output-dir", default="damaged_cans_dataset", 
                       help="Output directory for downloads")
    parser.add_argument("--urls-file", help="File containing URLs to download")
    parser.add_argument("--pexels-key", help="Pexels API key")
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = DamagedCanDownloader(args.output_dir)
    
    # Set API keys if provided
    if args.pexels_key:
        downloader.pexels_api_key = args.pexels_key
    
    # Run download process
    if args.urls_file:
        downloader.download_from_urls(args.urls_file)
    else:
        downloader.run()


if __name__ == "__main__":
    main()
