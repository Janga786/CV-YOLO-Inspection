import requests
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# The folder where you want to save the downloaded defect images.
DOWNLOAD_FOLDER = os.path.expanduser("~/YOLO_2/defect_textures")

# --- IMPORTANT: PASTE YOUR PEXELS API KEY HERE ---
# Get your free key from: https://www.pexels.com/api/new/
PEXELS_API_KEY = "NZVDUv6JNrJynNuyh2DhJAL14sEnx22ok29RnbiHzFz8qZeW9hUHHX22"

# --- SEARCH PARAMETERS ---
# Search terms specifically for defects that can be applied to coke cans
SEARCH_QUERIES = [
    # Scratches
    "metal scratches", "surface scratches", "scratch marks", "scratched metal",
    "aluminum scratches", "steel scratches", "brushed metal damage",
    
    # Dents
    "dented metal", "metal dents", "dent damage", "impact damage",
    "crushed metal", "bent metal", "deformed metal surface",
    
    # Punctures and holes
    "metal holes", "punctured metal", "bullet holes metal", "drill holes",
    "perforated metal", "damaged metal surface", "hole in metal",
    
    # General metal damage
    "damaged metal", "metal surface damage", "worn metal", "weathered metal",
    "rusted metal", "corroded metal", "aged metal surface",
    
    # Texture patterns that could represent defects
    "cracked surface", "damaged texture", "rough metal", "distressed metal",
    "metal wear patterns", "surface imperfections", "metal corrosion"
]

# Number of images to download per search query
IMAGES_PER_QUERY = 30

# Create subfolders for different defect types
DEFECT_CATEGORIES = {
    "scratches": ["metal scratches", "surface scratches", "scratch marks", "scratched metal",
                  "aluminum scratches", "steel scratches", "brushed metal damage"],
    "dents": ["dented metal", "metal dents", "dent damage", "impact damage",
              "crushed metal", "bent metal", "deformed metal surface"],
    "punctures": ["metal holes", "punctured metal", "bullet holes metal", "drill holes",
                  "perforated metal", "damaged metal surface", "hole in metal"],
    "general_damage": ["damaged metal", "metal surface damage", "worn metal", "weathered metal",
                       "rusted metal", "corroded metal", "aged metal surface",
                       "cracked surface", "damaged texture", "rough metal", "distressed metal",
                       "metal wear patterns", "surface imperfections", "metal corrosion"]
}


def get_defect_category(query):
    """Determine which defect category a search query belongs to."""
    for category, queries in DEFECT_CATEGORIES.items():
        if query in queries:
            return category
    return "general_damage"


def download_file(url, folder, filename):
    """Downloads a file from a URL into a specified folder with a given filename."""
    local_filename = os.path.join(folder, filename)

    if os.path.exists(local_filename):
        # Don't print a skip message to keep the output clean
        return True

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(local_filename, 'wb') as f, tqdm(
                desc=os.path.basename(local_filename),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                leave=False # Keep the progress bar on one line
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(local_filename):
            os.remove(local_filename)
        return False


def main():
    """Main function to find and download defect images using the Pexels API."""
    if PEXELS_API_KEY == "YOUR_API_KEY":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! FATAL ERROR: Please paste your Pexels API key    !!!")
        print("!!! into the PEXELS_API_KEY variable in the script.  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    print(f"Starting defect image download from Pexels API...")
    print(f"Saving files to: {DOWNLOAD_FOLDER}")
    
    # Create main download folder and subfolders for each defect category
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
    for category in DEFECT_CATEGORIES.keys():
        os.makedirs(os.path.join(DOWNLOAD_FOLDER, category), exist_ok=True)

    headers = {
        "Authorization": PEXELS_API_KEY
    }
    
    api_url = "https://api.pexels.com/v1/search"
    total_downloaded = 0
    downloads_by_category = {category: 0 for category in DEFECT_CATEGORIES.keys()}

    for query in SEARCH_QUERIES:
        print(f"\nSearching for '{query}'...")
        defect_category = get_defect_category(query)
        category_folder = os.path.join(DOWNLOAD_FOLDER, defect_category)
        
        params = {
            "query": query,
            "per_page": IMAGES_PER_QUERY,
            "orientation": "all"  # Allow all orientations for defect textures
        }
        
        try:
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            photos = data.get("photos", [])
            if not photos:
                print(f"No results found for '{query}'.")
                continue

            for photo in tqdm(photos, desc=f"Downloading '{query}' -> {defect_category}"):
                # We want the 'large' version of the image for better quality
                image_url = photo['src']['large']
                # Create a unique filename with category prefix
                filename = f"{defect_category}_pexels_{photo['id']}.jpg"
                
                if download_file(image_url, category_folder, filename):
                    total_downloaded += 1
                    downloads_by_category[defect_category] += 1

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for query '{query}': {e}")
            if "429" in str(e):
                print("Rate limit exceeded. Please wait a while before trying again.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred for query '{query}': {e}")
            continue

    print(f"\n--- Download process complete! ---")
    print(f"Successfully downloaded {total_downloaded} defect images.")
    print(f"\nBreakdown by category:")
    for category, count in downloads_by_category.items():
        print(f"  {category}: {count} images")
    
    print(f"\nNext steps:")
    print(f"1. Review the downloaded images in {DOWNLOAD_FOLDER}")
    print(f"2. Remove any images that don't look like usable defects")
    print(f"3. Use image composition techniques to apply these defects to your coke can images")
    print(f"4. Generate bounding box annotations for the applied defects")


if __name__ == "__main__":
    main()
