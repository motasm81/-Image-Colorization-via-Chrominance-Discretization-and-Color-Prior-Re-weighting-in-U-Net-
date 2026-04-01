import os
import numpy as np
from PIL import Image
from skimage import color
import torch
from torch.utils.data import Dataset

from config import IMG_SIZE, NUM_BINS

BIN_SIZE = 256 / NUM_BINS


class ColorizationDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = [
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])

        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_np = np.asarray(img, dtype=np.float32) / 255.0
        lab = color.rgb2lab(img_np)

        L = lab[:, :, 0] / 100.0
        a = np.clip(lab[:, :, 1], -128, 127)
        b = np.clip(lab[:, :, 2], -128, 127)

        a_bin = ((a + 128) / BIN_SIZE).astype(np.int64)
        b_bin = ((b + 128) / BIN_SIZE).astype(np.int64)

        a_bin = np.clip(a_bin, 0, NUM_BINS - 1)
        b_bin = np.clip(b_bin, 0, NUM_BINS - 1)

        class_map = a_bin * NUM_BINS + b_bin

        L = torch.from_numpy(L).unsqueeze(0).float()
        class_map = torch.from_numpy(class_map).long()

        return L, class_map



'''
import os
import numpy as np
from PIL import Image
from skimage import color
import torch
from torch.utils.data import Dataset
from config import IMG_SIZE, NUM_BINS

AB_RANGE = (-128, 127)
BIN_SIZE = (AB_RANGE[1] - AB_RANGE[0]) / NUM_BINS


def ab_to_bin(a, b):
    a_bin = int((a - AB_RANGE[0]) / BIN_SIZE)
    b_bin = int((b - AB_RANGE[0]) / BIN_SIZE)

    a_bin = np.clip(a_bin, 0, NUM_BINS - 1)
    b_bin = np.clip(b_bin, 0, NUM_BINS - 1)

    return a_bin * NUM_BINS + b_bin


class ColorizationDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])

        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        img_np = np.array(img) / 255.0
        lab = color.rgb2lab(img_np)

        L = lab[:, :, 0] / 100.0
        ab = lab[:, :, 1:3]

        a = ab[:, :, 0]
        b = ab[:, :, 1]

        a_bin = ((a + 128) / BIN_SIZE).astype(np.int64)
        b_bin = ((b + 128) / BIN_SIZE).astype(np.int64)

        a_bin = np.clip(a_bin, 0, NUM_BINS - 1)
        b_bin = np.clip(b_bin, 0, NUM_BINS - 1)

        class_map = a_bin * NUM_BINS + b_bin

        L = torch.tensor(L).unsqueeze(0).float()
        class_map = torch.tensor(class_map)

        return L, class_map

'''
