import requests
import os
import time
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup

class FlickrImageFinder:
    def __init__(self, output_dir="flickr_found_images_debug"):
        self.output_dir = Path(output_dir)
        self.damaged_cans_dir = self.output_dir / "damaged_cans"
        self.overlays_dir = self.output_dir / "overlays"
        self.scratches_dir = self.overlays_dir / "scratches"
        self.dents_dir = self.overlays_dir / "dents"
        self.punctures_dir = self.overlays_dir / "punctures"

        for dir_path in [self.damaged_cans_dir, self.scratches_dir, self.dents_dir, self.punctures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

    def download_file(self, url, filepath, description=""):
        try:
            response = self.session.get(url, stream=True, timeout=10)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True, desc=description) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False

    def search_flickr_images(self, query, num_images=10):
        urls = []
        page = 1
        print(f"Searching for '{query}' on Flickr...")
        while len(urls) < num_images:
            search_url = f"https://www.flickr.com/search/?text={query}&page={page}"
            try:
                response = self.session.get(search_url)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                img_tags = soup.find_all("img")
                print(f"Found {len(img_tags)} img tags on page {page}.")
                if not img_tags:
                    print("No more img tags found. Stopping search.")
                    break
                for img in img_tags:
                    if 'src' in img.attrs and 'staticflickr.com' in img['src']:
                        url = img['src']
                        print(f"Found image URL: {url}")
                        full_url = url.replace("_m.jpg", "_b.jpg")
                        urls.append(full_url)
                        if len(urls) >= num_images:
                            break
                page += 1
                time.sleep(1)
            except Exception as e:
                print(f"Error searching Flickr for {query}: {e}")
                break
        print(f"Found {len(urls)} image URLs for '{query}'.")
        return urls

    def find_and_download(self, queries, download_dir, target_count):
        downloaded_count = 0
        for query in queries:
            if downloaded_count >= target_count:
                break
            print(f"\nProcessing query: {query}")
            image_urls = self.search_flickr_images(query, num_images=target_count * 2)
            for i, url in enumerate(image_urls):
                if downloaded_count >= target_count:
                    break
                filename = f"{query.replace(' ', '_')}_{i}.jpg"
                filepath = download_dir / filename
                if not filepath.exists():
                    if self.download_file(url, filepath, f"Downloading {downloaded_count + 1}/{target_count}"):
                        downloaded_count += 1
        return downloaded_count

    def run(self, target_per_category=5):
        print("=== Starting Flickr Image Finder V2 (Debug Mode) ===")

        damaged_can_queries = [
            "dented can"
        ]
        print(f"\n--- Finding Damaged Cans ({target_per_category}) ---")
        self.find_and_download(damaged_can_queries, self.damaged_cans_dir, target_per_category)

        print("\n=== Flickr Image Finder V2 (Debug Mode) Finished ===")

if __name__ == '__main__':
    finder = FlickrImageFinder()
    finder.run()
