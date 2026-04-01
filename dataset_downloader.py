import os
import time
import requests
from tqdm import tqdm

# ================= SETTINGS =================
PIXABAY_API_KEY = "53852135-53641786095e264377dfb839a"

BASE_DIR = "dataset_pixabay"
CATEGORIES = {
    "animals": 1000

}

PER_PAGE = 100
SLEEP_TIME = 0.5  # be polite to API

# ================= DOWNLOAD FUNCTION =================
def download_pixabay(query, limit, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    downloaded = 0
    page = 1

    while downloaded < limit:
        url = (
            "https://pixabay.com/api/"
            f"?key={PIXABAY_API_KEY}"
            f"&q={query}"
            "&image_type=photo"
            "&orientation=horizontal"
            f"&per_page={PER_PAGE}"
            f"&page={page}"
        )

        response = requests.get(url, timeout=10)

        # ---- SAFETY CHECK ----
        if response.status_code != 200:
            print(f"[ERROR] HTTP {response.status_code}")
            print(response.text[:200])
            break

        try:
            data = response.json()
        except Exception:
            print("[ERROR] Invalid JSON response")
            print(response.text[:200])
            break

        hits = data.get("hits", [])
        if not hits:
            print("[INFO] No more images found.")
            break

        for img in hits:
            if downloaded >= limit:
                break

            img_url = img.get("largeImageURL")
            if not img_url:
                continue

            img_path = os.path.join(out_dir, f"{query}_{downloaded:05d}.jpg")

            try:
                img_data = requests.get(img_url, timeout=10).content
                with open(img_path, "wb") as f:
                    f.write(img_data)
                downloaded += 1
            except:
                continue

        print(f"[{query}] {downloaded}/{limit}")
        page += 1
        time.sleep(SLEEP_TIME)

# ================= MAIN =================
if __name__ == "__main__":
    for category, count in CATEGORIES.items():
        print(f"\nDownloading {category} images...")
        download_pixabay(
            query=category,
            limit=count,
            out_dir=os.path.join(BASE_DIR, category)
        )

    print("\n✅ Pixabay download complete")
