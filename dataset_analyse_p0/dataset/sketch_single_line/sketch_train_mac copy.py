import os
import random
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


# ==========================================
# Config
# ==========================================

TARGET_DIR = "../alien_tensors_raw"
CANVAS_SIZE = 256

NUM_STROKES = 12
NUM_SAMPLES = 60
EPOCHS = 1000

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 使用 MPS")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🚀 使用 CUDA")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ 使用 CPU")


# ==========================================
# Target glyph
# ==========================================

def get_random_glyph_target(font_dir, size=CANVAS_SIZE):

    font_files = glob.glob(os.path.join(font_dir, "*.[ot]tf"))
    if not font_files:
        raise ValueError("找不到字体文件")

    font_path = random.choice(font_files)
    font = ImageFont.truetype(font_path, int(size * 0.8))

    chars = [chr(c) for c in range(33, 126)] + [chr(c) for c in range(0x4E00, 0x4E80)]
    random.shuffle(chars)

    for ch in chars:

        try:
            bbox = font.getbbox(ch)
            if bbox is None:
                continue

            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            if w < 20:
                continue

            img = Image.new("L", (size, size), 255)
            draw = ImageDraw.Draw(img)

            draw.text(
                ((size - w) / 2, (size - h) / 2),
                ch,
                font=font,
                fill=0
            )

            arr = np.array(img).astype(np.float32) / 255.0
            arr = 1.0 - arr  # 黑色=1

            return torch.tensor(arr).to(DEVICE), font_path, ch

        except:
            continue

    raise RuntimeError("无法渲染字符")


# ==========================================
# Bézier Stroke (核心)
# ==========================================

class BezierStroke(nn.Module):

    def __init__(self):
        super().__init__()

        self.P = nn.Parameter(
            torch.rand(4, 2) * (CANVAS_SIZE * 0.5)
            + CANVAS_SIZE * 0.25
        )

        self.width = nn.Parameter(torch.tensor(2.5))
        self.alpha_logit = nn.Parameter(torch.tensor(2.0))

    def bezier(self, P, t):
        mt = 1 - t
        return (
            mt**3 * P[0]
            + 3 * mt**2 * t * P[1]
            + 3 * mt * t**2 * P[2]
            + t**3 * P[3]
        )

    def forward(self, grid_x, grid_y):

        t = torch.linspace(0, 1, NUM_SAMPLES, device=DEVICE).unsqueeze(1)

        curve = self.bezier(self.P, t)  # [T,2]

        cx = curve[:, 0]
        cy = curve[:, 1]

        dx = grid_x.unsqueeze(-1) - cx
        dy = grid_y.unsqueeze(-1) - cy

        dist2 = dx**2 + dy**2

        stroke = torch.exp(
            -dist2 / (2 * (self.width**2 + 1e-6))
        )

        img = torch.max(stroke, dim=-1)[0]

        alpha = torch.sigmoid(self.alpha_logit)

        return img * alpha


# ==========================================
# Render
# ==========================================

def render(strokes, grid_x, grid_y):

    img = torch.zeros_like(grid_x)

    for s in strokes:
        layer = s(grid_x, grid_y)

        # alpha compositing
        img = img + layer - img * layer

    return torch.clamp(img, 0, 1)


# ==========================================
# Loss
# ==========================================

def compute_loss(pred, target, strokes):

    # 1. IoU loss（核心）
    inter = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - inter
    loss_iou = 1.0 - (inter + 1e-6) / (union + 1e-6)

    # 2. Outside penalty（防止跑出结构）
    loss_outside = torch.mean(pred * (1.0 - target))

    # 3. sparsity（控制 stroke 数量）
    total_alpha = sum(torch.sigmoid(s.alpha_logit) for s in strokes)
    loss_sparse = total_alpha

    loss = (
        loss_iou * 10.0
        + loss_outside * 5.0
        + 0.01 * loss_sparse
    )

    return loss, loss_iou, loss_outside


# ==========================================
# Train
# ==========================================

def train():

    target, font_path, ch = get_random_glyph_target(TARGET_DIR)

    print("\nCharacter:", ch)
    print("Font:", font_path)

    x = torch.linspace(0, CANVAS_SIZE - 1, CANVAS_SIZE, device=DEVICE)
    y = torch.linspace(0, CANVAS_SIZE - 1, CANVAS_SIZE, device=DEVICE)

    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

    strokes = nn.ModuleList([
        BezierStroke().to(DEVICE)
        for _ in range(NUM_STROKES)
    ])

    optimizer = optim.Adam(strokes.parameters(), lr=0.01)

    for epoch in range(EPOCHS):

        optimizer.zero_grad()

        pred = render(strokes, grid_x, grid_y)

        loss, loss_iou, loss_outside = compute_loss(pred, target, strokes)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            for s in strokes:
                s.P.clamp_(0, CANVAS_SIZE)

        if epoch % 50 == 0:
            print(
                f"Epoch {epoch:04d} | "
                f"Loss {loss.item():.4f} | "
                f"IoU {loss_iou.item():.4f} | "
                f"Out {loss_outside.item():.4f}"
            )

    # ==========================================
    # Visualization
    # ==========================================

    pred = render(strokes, grid_x, grid_y)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(target.cpu(), cmap="gray")
    plt.title("Target")

    plt.subplot(1, 2, 2)
    plt.imshow(pred.detach().cpu(), cmap="gray")
    plt.title("Bezier Reconstruction")

    plt.show()

    # ==========================================
    # Output Bézier curves
    # ==========================================

    print("\n===== Bézier Stroke Output =====")

    for i, s in enumerate(strokes):
        P = s.P.detach().cpu().numpy()
        print(f"\nStroke {i}")
        print("P0:", P[0])
        print("P1:", P[1])
        print("P2:", P[2])
        print("P3:", P[3])


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":
    train()