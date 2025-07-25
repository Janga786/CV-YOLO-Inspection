import requests
import os
import time
from pathlib import Path
from tqdm import tqdm

class CCImageFinder:
    def __init__(self, output_dir="cc_found_images"):
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
            "User-Agent": "Gemini-CLI/1.0"
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
            # print(f"Error downloading {url}: {e}")
            if filepath.exists():
                filepath.unlink()
            return False

    def search_cc_images(self, query, num_images=10):
        urls = []
        page = 1
        while len(urls) < num_images:
            search_url = f"https://api.creativecommons.engineering/v1/works?q={query}&page_size=50&page={page}"
            try:
                response = self.session.get(search_url)
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
                if not results:
                    break
                for item in results:
                    if item.get("url"):
                        urls.append(item["url"])
                        if len(urls) >= num_images:
                            break
                page += 1
                time.sleep(1)
            except Exception as e:
                print(f"Error searching Creative Commons for {query}: {e}")
                break
        return urls

    def find_and_download(self, queries, download_dir, target_count):
        downloaded_count = 0
        for query in queries:
            if downloaded_count >= target_count:
                break
            print(f"\nSearching for: {query}")
            image_urls = self.search_cc_images(query, num_images=target_count * 2)
            for i, url in enumerate(image_urls):
                if downloaded_count >= target_count:
                    break
                filename = f"{query.replace(' ', '_')}_{i}.jpg"
                filepath = download_dir / filename
                if not filepath.exists():
                    if self.download_file(url, filepath, f"Downloading {downloaded_count + 1}/{target_count}"):
                        downloaded_count += 1
        return downloaded_count

    def run(self, target_per_category=50):
        print("=== Starting Creative Commons Image Finder ===")

        damaged_can_queries = [
            "dented can", "crushed can", "damaged can",
            "scratched can", "punctured can"
        ]
        print(f"\n--- Finding Damaged Cans ({target_per_category}) ---")
        self.find_and_download(damaged_can_queries, self.damaged_cans_dir, target_per_category)

        scratch_queries = ["scratch overlay", "metal scratch texture"]
        print(f"\n--- Finding Scratch Overlays ({target_per_category}) ---")
        self.find_and_download(scratch_queries, self.scratches_dir, target_per_category)

        dent_queries = ["dent overlay", "metal dent texture"]
        print(f"\n--- Finding Dent Overlays ({target_per_category}) ---")
        self.find_and_download(dent_queries, self.dents_dir, target_per_category)

        puncture_queries = ["puncture overlay", "bullet hole texture"]
        print(f"\n--- Finding Puncture Overlays ({target_per_category}) ---")
        self.find_and_download(puncture_queries, self.punctures_dir, target_per_category)

        print("\n=== Creative Commons Image Finder Finished ===")

if __name__ == '__main__':
    finder = CCImageFinder()
    finder.run()
