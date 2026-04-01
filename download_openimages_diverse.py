import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil

# ================= SETTINGS =================
INPUT_DIR = r"C:\Users\user\PycharmProjects\Colorization_Proj\dataset2\landscape"
OUTPUT_DIR = r"C:\Users\user\PycharmProjects\Colorization_Proj\dataset2\ADDon_clean"

# VERY LENIENT thresholds
MIN_COLOR_STD = 1.5        # below this = almost grayscale
MIN_SATURATION = 2.0       # HSV saturation (0–255)

MAX_IMAGES = None
LOG_EVERY = 500
# ===========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_very_weak_color(img):
    """
    Returns True ONLY if image is essentially grayscale
    """
    if img.ndim != 3 or img.shape[2] != 3:
        return True

    # RGB channel difference
    b, g, r = cv2.split(img)
    color_std = np.mean([
        np.std(r - g),
        np.std(r - b),
        np.std(g - b)
    ])

    if color_std < MIN_COLOR_STD:
        return True

    # HSV saturation check
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    if hsv[:, :, 1].mean() < MIN_SATURATION:
        return True

    return False


def scan_images(root):
    exts = (".jpg", ".jpeg", ".png")
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in exts:
            yield p


imgs = list(scan_images(INPUT_DIR))
print(f"🔍 Found {len(imgs)} images total")
print("🧹 Removing only grayscale / very weak-color images...")

saved = 0
removed = 0

for img_path in tqdm(imgs):
    if MAX_IMAGES and saved >= MAX_IMAGES:
        break

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    if is_very_weak_color(img):
        removed += 1
        continue

    out_path = os.path.join(OUTPUT_DIR, f"img_{saved:07d}.jpg")
    shutil.copy(img_path, out_path)
    saved += 1

    if saved % LOG_EVERY == 0:
        print(f"✅ Saved {saved} images")

print("\n🎉 DONE")
print(f"Saved images   : {saved}")
print(f"Removed images : {removed}")
print(f"Kept ratio     : {saved / (saved + removed + 1e-6):.2%}")
