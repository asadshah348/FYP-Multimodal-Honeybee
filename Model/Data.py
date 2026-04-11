import os
import requests

# -------- CONFIG --------
ACCESS_KEY = "IgnDcpIcBI7G3tf5MRePhtw0IJs7ZoITozTc05byeIYOuREqzyDrXFVp"  # Replace with your Pexels API key
QUERY = "bees on beehive"
TOTAL_IMAGES = 5000  # Total images you want after resuming
SAVE_DIR = "bee_images2"
PER_PAGE = 80  # Max per Pexels API
# ------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

url = "https://api.pexels.com/v1/search"
headers = {
    "Authorization": ACCESS_KEY
}

# Get list of already downloaded images
existing_files = set(os.listdir(SAVE_DIR))
downloaded = len(existing_files)
page = (downloaded // PER_PAGE) + 1  # Resume from the right page

print(f"Resuming download: already have {downloaded} images, starting from page {page}")

while downloaded < TOTAL_IMAGES:
    params = {
        "query": QUERY,
        "per_page": PER_PAGE,
        "page": page
    }

    response = requests.get(url, headers=headers, params=params)
    data = response.json()

    if "photos" not in data or len(data["photos"]) == 0:
        print("No more images found!")
        break

    for item in data["photos"]:
        if downloaded >= TOTAL_IMAGES:
            break

        image_url = item["src"]["original"]
        filename = image_url.split("/")[-1].split("?")[0]

        # Skip if already downloaded
        if filename in existing_files:
            print(f"Skipped (already exists): {filename}")
            continue

        try:
            img_data = requests.get(image_url, timeout=15).content
            with open(os.path.join(SAVE_DIR, filename), "wb") as f:
                f.write(img_data)
            downloaded += 1
            existing_files.add(filename)
            print(f"Downloaded {downloaded}: {filename}")
        except Exception as e:
            print(f"Failed to download {image_url}: {e}")

    page += 1

print(f"Done! {downloaded} images downloaded in '{SAVE_DIR}' folder ✅")
