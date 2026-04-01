import os
import glob
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

FINE_TUNE_LR = 2e-3
EPOCHS = 24

USE_COLOR_PRIOR = True
PRIOR_ALPHA = 0.4
USE_EMA = True

CHECKPOINT_DIR = "checkpoints_finetune_addon_TF"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DATASET_PATH = r"C:\Users\user\PycharmProjects\Colorization_Proj\dataset2\ADDon_clean"

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

    start_epoch = 0
    optimizer_state = None
    scheduler_state = None
    ema_state = None

    # ------------------ LOAD LAST CHECKPOINT IF EXISTS ------------------
    ckpt_files = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth")))
    if ckpt_files:
        last_ckpt = ckpt_files[-1]
        ckpt_data = torch.load(last_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt_data["model_state_dict"])
        start_epoch = ckpt_data.get("epoch", 0) + 1
        optimizer_state = ckpt_data.get("optimizer_state_dict", None)
        scheduler_state = ckpt_data.get("scheduler_state_dict", None)
        ema_state = ckpt_data.get("ema_state_dict", None)
        print(f"Resuming from checkpoint {last_ckpt}, starting at epoch {start_epoch}")
    else:
        # Load pretrained model if no checkpoint
        pretrained_ckpt = torch.load("checkpoints/color_cnn_epoch_56.pth", map_location=DEVICE)
        model.load_state_dict(pretrained_ckpt["model_state_dict"])
        print("Loaded pretrained model")

    # ------------------ LOSS ------------------
    if USE_COLOR_PRIOR:
        prior = np.load("color_prior_T6.npy")
        weights = 1.0 / (prior ** PRIOR_ALPHA + 1e-6)
        weights /= weights.mean()
        weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=weights)
        print("Using original color prior")
    else:
        criterion = nn.CrossEntropyLoss()

    # ------------------ OPTIMIZER ------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=FINE_TUNE_LR,
        weight_decay=1e-4
    )
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    # ------------------ SCHEDULER ------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)

    # ------------------ EMA ------------------
    ema = None
    if USE_EMA:
        from torch_ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        if ema_state:
            ema.load_state_dict(ema_state)
        print("EMA enabled")

    # ------------------ TRAIN LOOP ------------------
    for epoch in range(start_epoch, start_epoch + EPOCHS):
        model.train()
        epoch_loss = 0.0

        for L, target in tqdm(loader, desc=f"Fine-tune Epoch {epoch+1}/{start_epoch+EPOCHS}"):
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

        scheduler.step(avg_loss)

        # ------------------ SAVE ------------------
        save_path = os.path.join(
            CHECKPOINT_DIR,
            f"color_cnn_addon_epoch_{epoch+1}.pth"
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
