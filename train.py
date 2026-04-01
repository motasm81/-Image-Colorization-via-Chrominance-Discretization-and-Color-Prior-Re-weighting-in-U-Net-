import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import ColorizationDataset
from model import UNetColorCNN


# ------------------ SAFETY (Windows) ------------------
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()


# ------------------ SETTINGS ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_COLOR_PRIOR = True
PRIOR_ALPHA = 0.3
USE_EMA = True

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ------------------ UTILS ------------------
def get_latest_checkpoint():
    ckpts = [
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("color_cnn_epoch_") and f.endswith(".pth")
    ]
    if not ckpts:
        return None

    ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(CHECKPOINT_DIR, ckpts[-1])


# ------------------ MAIN ------------------
def main():
    print("Using device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ------------------ DATA ------------------
    dataset = ColorizationDataset(DATASET_PATH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True
    )

    # ------------------ MODEL ------------------
    model = UNetColorCNN(NUM_BINS).to(DEVICE)

    # ------------------ LOSS ------------------
    if USE_COLOR_PRIOR:
        prior = np.load("color_prior.npy")
        weights = 1.0 / (prior ** PRIOR_ALPHA + 1e-6)
        weights /= weights.mean()
        weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Using color prior reweighting")
    else:
        criterion = nn.CrossEntropyLoss()

    # ------------------ OPTIMIZER ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ------------------ SCHEDULER ------------------
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[40, 60],
        gamma=0.3
    )

    # ------------------ EMA ------------------
    ema = None
    if USE_EMA:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        print("EMA enabled")

    # ------------------ RESUME ------------------
    start_epoch = 0
    latest_ckpt = get_latest_checkpoint()

    if latest_ckpt:
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        if ema and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])

        start_epoch = ckpt["epoch"] + 1

        # CRITICAL: sync scheduler epoch
        scheduler.last_epoch = start_epoch - 1
        print(f"Resumed from epoch {start_epoch}")

        # Late-stage fine-tuning safeguard
        if start_epoch >= 29:
            for g in optimizer.param_groups:
                g["lr"] = 2e-3
                print(f"Param group : LR set to {g['lr']:.8f}")

    # ------------------ TRAIN LOOP ------------------
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        for L, target in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            L = L.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(L)
            loss = criterion(logits, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if ema:
                ema.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(
            f"Epoch {epoch+1} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.8f}"
        )

        # MultiStepLR step
        scheduler.step()

        # ------------------ SAVE ------------------
        save_path = os.path.join(
            CHECKPOINT_DIR,
            f"color_cnn_epoch_{epoch+1}.pth"
        )

        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

        if ema:
            save_dict["ema_state_dict"] = ema.state_dict()

        torch.save(save_dict, save_path)
        print(f"Saved {save_path}")


# ------------------ RUN ------------------
if __name__ == "__main__":
    main()
















'''


import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from dataset import ColorizationDataset
from model import UNetColorCNN

# ------------------ SAFETY (Windows) ------------------
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()

# ------------------ SETTINGS ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_COLOR_PRIOR = True
PRIOR_ALPHA = 0.3
USE_EMA = True

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ------------------ UTILS ------------------
def get_latest_checkpoint():
    ckpts = [
        f for f in os.listdir(CHECKPOINT_DIR)
        if f.startswith("color_cnn_epoch_") and f.endswith(".pth")
    ]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(CHECKPOINT_DIR, ckpts[-1])

# ------------------ MAIN ------------------
def main():
    print("Using device:", DEVICE)
    if DEVICE == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # ------------------ DATA ------------------
    dataset = ColorizationDataset(DATASET_PATH)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,      # Windows-safe
        pin_memory=True,
        persistent_workers=True
    )

    # ------------------ MODEL ------------------
    model = UNetColorCNN(NUM_BINS).to(DEVICE)

    # ------------------ LOSS ------------------
    if USE_COLOR_PRIOR:
        prior = np.load("color_prior.npy")
        weights = 1.0 / (prior ** PRIOR_ALPHA + 1e-6)
        weights /= weights.mean()
        weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Using color prior reweighting")
    else:
        criterion = nn.CrossEntropyLoss()

    # ------------------ OPTIMIZER ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 🔥 COCO-CORRECT scheduler (THIS WAS THE BIG FIX)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",factor=0.3,patience=2,  threshold=2e-4,       threshold_mode="abs",min_lr=1e-5,verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=1,
        threshold=1e-4,
        threshold_mode="abs",
        min_lr=1e-5,
    )

    # ------------------ EMA ------------------
    ema = None
    if USE_EMA:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        print("EMA enabled")

    # ------------------ RESUME ------------------
    start_epoch = 0
    latest_ckpt = get_latest_checkpoint()
    if latest_ckpt:
        ckpt = torch.load(latest_ckpt, map_location=DEVICE)

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            print("No scheduler state found in checkpoint (old checkpoint)")
        # 🔥 IMPORTANT

        start_epoch = ckpt["epoch"] + 1

        if ema and "ema_state_dict" in ckpt:
            ema.load_state_dict(ckpt["ema_state_dict"])

        print(f"Resumed from epoch {start_epoch}")

    # ------------------ TRAIN LOOP ------------------
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        for L, target in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            L = L.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(L)
            loss = criterion(logits, target)
            loss.backward()

            # 🔥 Stabilizes COCO training
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            if ema:
                ema.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        print(
            f"Epoch {epoch + 1} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.8f}"
        )

        scheduler.step(avg_loss)

        # ------------------ SAVE ------------------
        save_path = os.path.join(
            CHECKPOINT_DIR,
            f"color_cnn_epoch_{epoch+1}.pth"
        )

        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),  # 🔥 REQUIRED
        }

        if ema:
            save_dict["ema_state_dict"] = ema.state_dict()

        torch.save(save_dict, save_path)
        print(f"Saved {save_path}")

# ------------------ RUN ------------------
if __name__ == "__main__":
    main()




import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm

from config import *
from dataset import ColorizationDataset
from model import UNetColorCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_COLOR_PRIOR = True
PRIOR_ALPHA = 0.3
USE_EMA = True

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_latest_checkpoint():
    ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(CHECKPOINT_DIR, ckpts[-1])

def main():
    dataset = ColorizationDataset(DATASET_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = UNetColorCNN(NUM_BINS).to(DEVICE)

    if USE_COLOR_PRIOR:
        prior = np.load("color_prior.npy")
        weights = 1.0 / (prior ** PRIOR_ALPHA + 1e-6)
        weights = weights / weights.mean()
        weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.3,
        patience=2,  # COCO converges slower but needs earlier decay
        min_lr=1e-5
    )

    if USE_EMA:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)

    start_epoch = 0
    latest_ckpt = get_latest_checkpoint()
    if latest_ckpt:
        checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        if USE_EMA and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0

        for L, target in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            L, target = L.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            logits = model(L)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            if USE_EMA:
                ema.update()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")
        scheduler.step(avg_loss)

        save_path = os.path.join(CHECKPOINT_DIR, f"color_cnn_epoch_{epoch+1}.pth")
        save_dict = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }
        if USE_EMA:
            save_dict["ema_state_dict"] = ema.state_dict()
        torch.save(save_dict, save_path)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()

'''
