import numpy as np
import os
from PIL import Image
from skimage import color
from tqdm import tqdm


class ColorPriorGenerator:
    def __init__(self, dataset_path, num_bins=16, img_size=224):
        self.dataset_path = dataset_path
        self.num_bins = num_bins
        self.img_size = img_size

        self.counts = np.zeros(num_bins * num_bins, dtype=np.float64)
        self.bin_size = 256 / num_bins

    def _process_image(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return

        img = img.resize((self.img_size, self.img_size))
        img_np = np.asarray(img, dtype=np.float32) / 255.0

        lab = color.rgb2lab(img_np)
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        a_bin = np.clip(
            ((a + 128) / self.bin_size).astype(np.int32),
            0,
            self.num_bins - 1
        )
        b_bin = np.clip(
            ((b + 128) / self.bin_size).astype(np.int32),
            0,
            self.num_bins - 1
        )

        idx = a_bin * self.num_bins + b_bin

        # vectorized histogram update
        bincount = np.bincount(
            idx.flatten(),
            minlength=self.num_bins * self.num_bins
        )
        self.counts += bincount

    def compute_prior(self):
        image_paths = []
        for root, _, files in os.walk(self.dataset_path):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_paths.append(os.path.join(root, f))

        print(f"Found {len(image_paths)} images")

        for img_path in tqdm(image_paths, desc="Computing color prior"):
            self._process_image(img_path)

        # avoid zero bins
        self.counts += 1e-6
        self.counts /= self.counts.sum()

        return self.counts

    def save_prior(self, file_path="color_prior.npy"):
        np.save(file_path, self.counts)
        print(f"Saved color prior to {file_path}")
        print("Prior shape:", self.counts.shape)
        print("Min:", self.counts.min(), "Max:", self.counts.max())


# ------------------ RUN ------------------
if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\user\PycharmProjects\Colorization_Proj\dataset2\coco_clean_color"
    NUM_BINS = 16
    IMG_SIZE = 224

    generator = ColorPriorGenerator(
        dataset_path=DATASET_PATH,
        num_bins=NUM_BINS,
        img_size=IMG_SIZE
    )

    prior = generator.compute_prior()
    generator.save_prior("color_prior.npy")




















'''















import numpy as np
import os
from PIL import Image
from skimage import color
from tqdm import tqdm

class ColorPriorGenerator:
    def __init__(self, dataset_path, num_bins=16, img_size=224):
        self.dataset_path = dataset_path
        self.num_bins = num_bins
        self.img_size = img_size
        self.counts = np.zeros((num_bins * num_bins), dtype=np.float64)

    def _process_image(self, img_path):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return
        img = img.resize((self.img_size, self.img_size))
        img_np = np.array(img) / 255.0
        lab = color.rgb2lab(img_np)
        ab = lab[:, :, 1:]  # a,b channels

        a = ab[:, :, 0]
        b = ab[:, :, 1]

        bin_size = 256 / self.num_bins
        a_bin = np.clip(((a + 128) / bin_size).astype(int), 0, self.num_bins - 1)
        b_bin = np.clip(((b + 128) / bin_size).astype(int), 0, self.num_bins - 1)

        idx = a_bin * self.num_bins + b_bin
        for i in idx.flatten():
            self.counts[i] += 1

    def compute_prior(self):
        images = os.listdir(self.dataset_path)
        for img_name in tqdm(images, desc="Computing color prior"):
            self._process_image(os.path.join(self.dataset_path, img_name))

        self.counts /= self.counts.sum()  # normalize
        return self.counts

    def save_prior(self, file_path="color_prior.npy"):
        np.save(file_path, self.counts)
        print(f"Saved color prior to {file_path}, shape: {self.counts.shape}")


if __name__ == "__main__":
    DATASET_PATH = r"C:\DataSets\flickr30k_images\flickr30k_image"
    NUM_BINS = 16
    IMG_SIZE = 224

    generator = ColorPriorGenerator(DATASET_PATH, NUM_BINS, IMG_SIZE)
    prior = generator.compute_prior()
    generator.save_prior("color_prior.npy")

import numpy as np
import os
from PIL import Image
from skimage import color
from tqdm import tqdm

def compute_color_prior(dataset_path, num_bins, img_size=224):
    counts = np.zeros((num_bins * num_bins), dtype=np.float64)

    images = os.listdir(dataset_path)

    for img_name in tqdm(images, desc="Computing color prior"):
        img_path = os.path.join(dataset_path, img_name)
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            continue  # skip invalid files

        img = img.resize((img_size, img_size))
        img_np = np.array(img) / 255.0

        lab = color.rgb2lab(img_np)
        ab = lab[:, :, 1:]  # a,b channels

        a = ab[:, :, 0]
        b = ab[:, :, 1]

        # Correct binning for [-128, 127] range
        bin_size = 256 / num_bins
        a_bin = np.clip(((a + 128) / bin_size).astype(int), 0, num_bins - 1)
        b_bin = np.clip(((b + 128) / bin_size).astype(int), 0, num_bins - 1)

        # Flatten and count
        idx = a_bin * num_bins + b_bin
        for i in idx.flatten():
            counts[i] += 1

    counts /= counts.sum()  # normalize to sum=1
    return counts


if __name__ == "__main__":
    NUM_BINS = 16  # <-- use 16x16 bins
    DATASET_PATH = r"C:\DataSets\flickr30k_images\flickr30k_image"

    print("Computing color prior...")
    prior = compute_color_prior(DATASET_PATH, NUM_BINS)
    np.save("color_prior.npy", prior)
    print(f"Saved color_prior.npy with shape {prior.shape}")
'''
