import requests
import os
from tqdm import tqdm

# --- CONFIGURATION ---
# The folder where you want to save the downloaded background images.
DOWNLOAD_FOLDER = os.path.expanduser("~/YOLO_2/hdri_backgrounds")

# --- IMPORTANT: PASTE YOUR PEXELS API KEY HERE ---
# Get your free key from: https://www.pexels.com/api/new/
PEXELS_API_KEY = "NZVDUv6JNrJynNuyh2DhJAL14sEnx22ok29RnbiHzFz8qZeW9hUHHX22"

# --- SEARCH PARAMETERS ---
# A list of search terms to get a wide variety of backgrounds
SEARCH_QUERIES = [
    "nature", "outdoor", "city street", "studio background", 
    "texture", "abstract", "office interior", "warehouse", "blurry background"
]

# Number of images to download per search query
IMAGES_PER_QUERY = 50


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
    """Main function to find and download images using the Pexels API."""
    if PEXELS_API_KEY == "YOUR_API_KEY":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! FATAL ERROR: Please paste your Pexels API key    !!!")
        print("!!! into the PEXELS_API_KEY variable in the script.  !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return

    print(f"Starting background image download from Pexels API...")
    print(f"Saving files to: {DOWNLOAD_FOLDER}")
    
    os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

    headers = {
        "Authorization": PEXELS_API_KEY
    }
    
    api_url = "https://api.pexels.com/v1/search"
    total_downloaded = 0

    for query in SEARCH_QUERIES:
        print(f"\nSearching for '{query}'...")
        params = {
            "query": query,
            "per_page": IMAGES_PER_QUERY,
            "orientation": "landscape"
        }
        
        try:
            response = requests.get(api_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            photos = data.get("photos", [])
            if not photos:
                print(f"No results found for '{query}'.")
                continue

            for photo in tqdm(photos, desc=f"Downloading '{query}'"):
                # We want the 'large' version of the image
                image_url = photo['src']['large']
                # Create a unique filename
                filename = f"pexels_{photo['id']}.jpg"
                
                if download_file(image_url, DOWNLOAD_FOLDER, filename):
                    total_downloaded += 1

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for query '{query}': {e}")
            if "429" in str(e):
                print("Rate limit exceeded. Please wait a while before trying again.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred for query '{query}': {e}")
            continue

    print(f"\n--- Download process complete! ---")
    print(f"Successfully downloaded {total_downloaded} new background images.")


if __name__ == "__main__":
    main()

