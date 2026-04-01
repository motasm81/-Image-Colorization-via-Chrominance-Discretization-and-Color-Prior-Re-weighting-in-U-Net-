import os
import cv2
from tqdm import tqdm
import shutil

IMG_SIZE = 256
DATASET_PATH = r"C:\Users\user\PycharmProjects\Colorization_Proj\dataset2\coco_clean_color"
REMOVED_DIR = os.path.join(DATASET_PATH, "removed")
os.makedirs(REMOVED_DIR, exist_ok=True)

tiny_images = []

# Collect all image files
all_files = []
for root, _, files in os.walk(DATASET_PATH):
    for f in files:
        if f.lower().endswith((".jpg", ".jpeg", ".png")):
            all_files.append(os.path.join(root, f))

print(f"Scanning {len(all_files)} images...")

# Check image sizes
for path in tqdm(all_files, desc="Checking image sizes"):
    img = cv2.imread(path)
    if img is None:
        continue
    h, w, _ = img.shape
    if min(h, w) < IMG_SIZE:
        tiny_images.append((path, w, h))
        # Move to removed folder
        shutil.move(path, REMOVED_DIR)

print(f"Removed {len(tiny_images)} images smaller than {IMG_SIZE}x{IMG_SIZE}")
for path, w, h in tiny_images[:20]:  # show first 20
    print(f"{path} -> {w}x{h}")






