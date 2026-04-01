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
