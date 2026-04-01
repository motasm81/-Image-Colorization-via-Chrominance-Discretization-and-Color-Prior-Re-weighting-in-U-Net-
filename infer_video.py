import cv2
import torch
import numpy as np
from config import NUM_BINS
from model import UNetColorCNN
from torch.cuda.amp import autocast

# ================= SETTINGS =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_PATH = r"C:\Users\user\PycharmProjects\Colorization_Proj\checkpoints_finetune_addon\color_cnn_addon_epoch_16.pth"
VIDEO_IN = r"C:\Users\user\Downloads\Nature Is Speaking Shailene Woodley is Forest - Conservation International (1080p, h264).mp4"
VIDEO_OUT = r"Forest2E162_T.6_C1.2_AB55.mp4"

IMG_SIZE = 256
BATCH_SIZE = 30

TEMPERATURE = .6
CHROMA_GAIN = 1.2
MAX_AB = 55
TEMPORAL_ALPHA = 0.75

# ================= LOAD MODEL =================
model = UNetColorCNN(NUM_BINS).to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ================= BIN CENTERS =================
BIN_SIZE = 256 / NUM_BINS

a_centers = torch.linspace(-128 + BIN_SIZE / 2, 128 - BIN_SIZE / 2, NUM_BINS, device=DEVICE)
b_centers = torch.linspace(-128 + BIN_SIZE / 2, 128 - BIN_SIZE / 2, NUM_BINS, device=DEVICE)

ab_centers = torch.stack(
    torch.meshgrid(a_centers, b_centers, indexing="ij"),
    dim=-1
).reshape(-1, 2)

# ================= VIDEO IO =================
cap = cv2.VideoCapture(VIDEO_IN)
fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter(
    VIDEO_OUT,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (W, H)
)

prev_a, prev_b = None, None

# ================= MAIN LOOP =================
while True:
    frames = []
    L_batch = []

    # ---------- READ BATCH ----------
    for _ in range(BATCH_SIZE):
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        small = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        lab_small = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_batch.append(lab_small[..., 0] / 255.0)

    if len(frames) == 0:
        break

    L_tensor = torch.from_numpy(np.stack(L_batch)).unsqueeze(1).float().to(DEVICE)

    # ---------- MODEL ----------
    with torch.no_grad():
        with autocast():
            logits = model(L_tensor)

        # IMPORTANT: FP32 for softmax & expectation
        logits = logits.float()
        probs = torch.softmax(logits / TEMPERATURE, dim=1)

    probs = probs.permute(0, 2, 3, 1)
    ab = torch.matmul(probs, ab_centers)

    a_batch = ab[..., 0].cpu().numpy()
    b_batch = ab[..., 1].cpu().numpy()

    # ---------- POST PROCESS EACH FRAME ----------
    for i, frame in enumerate(frames):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        # OpenCV LAB scale — DO NOT TOUCH L
        L_orig = lab[..., 0]

        a = a_batch[i]
        b = b_batch[i]

        # Upscale FIRST
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(b, (W, H), interpolation=cv2.INTER_CUBIC)

        # Temporal smoothing at full resolution
        if prev_a is not None:
            a = TEMPORAL_ALPHA * a + (1 - TEMPORAL_ALPHA) * prev_a
            b = TEMPORAL_ALPHA * b + (1 - TEMPORAL_ALPHA) * prev_b

        prev_a, prev_b = a.copy(), b.copy()

        # Chroma boost
        a *= CHROMA_GAIN
        b *= CHROMA_GAIN

        a = np.clip(a, -MAX_AB, MAX_AB)
        b = np.clip(b, -MAX_AB, MAX_AB)

        # Rebuild LAB (OpenCV expects 0–255)
        lab[..., 0] = L_orig
        lab[..., 1] = np.clip(a + 128, 0, 255)
        lab[..., 2] = np.clip(b + 128, 0, 255)

        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Light sharpen
        out = cv2.addWeighted(
            out, 1.05,
            cv2.GaussianBlur(out, (0, 0), 1),
            -0.05, 0
        )

        writer.write(out)

cap.release()
writer.release()
print("✅ FIXED video colorization complete")

