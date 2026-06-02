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
from scipy.ndimage import distance_transform_edt

TARGET_DIR = "../alien_tensors_raw"
CANVAS_SIZE = 256
NUM_STROKES = 12
NUM_SAMPLES = 60

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def get_random_glyph_target(font_dir, size=CANVAS_SIZE):
    font_files = glob.glob(os.path.join(font_dir, "*.[ot]tf"))
    if not font_files: raise ValueError("找不到字体")
    font_path = random.choice(font_files)
    pil_font = ImageFont.truetype(font_path, int(size * 0.8))
    
    valid_chars = [chr(c) for c in range(33, 126)] + [chr(c) for c in range(0x4E00, 0x4E50)]
    random.shuffle(valid_chars)
    for char in valid_chars:
        try:
            bbox = pil_font.getbbox(char)
            if bbox and (bbox[2]-bbox[0]) > 20:
                img = Image.new('L', (size, size), 255) 
                draw = ImageDraw.Draw(img)
                draw.text(((size-(bbox[2]-bbox[0]))/2, (size-(bbox[3]-bbox[1]))/2), char, font=pil_font, fill=0)
                target = torch.from_numpy(np.array(img)).float() / 255.0
                return target.unsqueeze(-1).repeat(1, 1, 4).to(DEVICE), font_path, char
        except: continue
    raise ValueError("未找到可渲染字符")


class GaussianStroke(nn.Module):
    def __init__(self, init_points):
        super().__init__()
        # 🌟 核心修改 1：精准空投！直接出生在字体的像素上
        idx = torch.randperm(init_points.shape[0])[:4]
        self.P = nn.Parameter(init_points[idx]) 
        
        # 宽度曲线：初始化为适中的固定宽度 (比如 6 个像素)
        self.Q = nn.Parameter(torch.tensor([[0.0, 0.0], [5.0, 6.0], [15.0, 6.0], [20.0, 0.0]]))
        
        # 存在概率：初始值极高(>99%)，杜绝摆烂
        self.alpha_logit = nn.Parameter(torch.tensor(5.0)) 

    def cubic_bezier(self, pts, t):
        mt = 1 - t
        return mt**3*pts[0] + 3*mt**2*t*pts[1] + 3*mt*t**2*pts[2] + t**3*pts[3]

    def forward(self, grid_x, grid_y):
        t = torch.linspace(0, 1, NUM_SAMPLES, device=DEVICE).unsqueeze(1)
        M_t = self.cubic_bezier(self.P, t)
        
        W_t = self.cubic_bezier(self.Q, t)
        base = self.P[3] - self.P[0] 
        radii = torch.abs(torch.matmul(W_t - self.Q[0], torch.stack([-base[1], base[0]]) / (torch.norm(base)+1e-6)))
        
        dist_sq = (grid_x.unsqueeze(-1) - M_t[:, 0])**2 + (grid_y.unsqueeze(-1) - M_t[:, 1])**2
        
        # 🌟 核心修改 2：使用 sum 代替 max，释放所有采样点的梯度！
        img_sum = torch.sum(torch.exp(-dist_sq / (radii.unsqueeze(0)**2 + 1e-2)), dim=-1)
        img = torch.clamp(img_sum, 0, 1) # 防止发光过度
        
        return img * torch.sigmoid(self.alpha_logit)


def train_pure_pytorch():
    target_img, _, char = get_random_glyph_target(TARGET_DIR)
    target_mask = (1.0 - target_img[:,:,0]).to(DEVICE)
    
    # 提取目标图像中所有为黑色的像素坐标，用于初始化“空投”
    y_idx, x_idx = torch.where(target_mask > 0.5)
    valid_points = torch.stack([x_idx, y_idx], dim=1).float()
    
    bg_mask_np = (target_img[:,:,0] > 0.5).cpu().numpy() 
    dt_map_np = distance_transform_edt(bg_mask_np)
    dt_map_np = dt_map_np / (np.max(dt_map_np) + 1e-6)
    gravity_field = torch.tensor(dt_map_np, dtype=torch.float32, device=DEVICE)

    x = torch.linspace(0, CANVAS_SIZE, CANVAS_SIZE, device=DEVICE)
    y = torch.linspace(0, CANVAS_SIZE, CANVAS_SIZE, device=DEVICE)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

    strokes = nn.ModuleList([GaussianStroke(valid_points).to(DEVICE) for _ in range(NUM_STROKES)])
    # 使用较小的学习率，让模型慢慢理顺打结的线条
    optimizer = optim.Adam(strokes.parameters(), lr=0.5)
    
    print(f"🎯 训练字符: {char} | 设备: {DEVICE}")

    for epoch in range(1500):
        optimizer.zero_grad()
        
        stroke_imgs = []
        total_alpha = 0
        for s in strokes:
            # 🌟 核心修改 3：前 300 轮冻结 alpha 的梯度，强迫它去扭曲曲线，而不是降 alpha 隐身
            if epoch < 300:
                s.alpha_logit.requires_grad = False
            else:
                s.alpha_logit.requires_grad = True
                
            img = s(grid_x, grid_y)
            stroke_imgs.append(img)
            total_alpha += torch.sigmoid(s.alpha_logit)
            
        stroke_imgs_tensor = torch.stack(stroke_imgs)
        pred_img = torch.clamp(torch.sum(stroke_imgs_tensor, dim=0), 0, 1)
        
        # ==================== 🌟 终极 Loss 组合 ====================
        
        # 1. Soft-IoU Loss & MSE Loss
        intersection = torch.sum(pred_img * target_mask)
        union = torch.sum(pred_img) + torch.sum(target_mask) - intersection
        loss_iou = 1.0 - (intersection + 1e-5) / (union + 1e-5)
        loss_mse = F.mse_loss(pred_img, target_mask)
        
        # 2. Gravity Loss
        loss_gravity = torch.sum(pred_img * gravity_field) / (CANVAS_SIZE * CANVAS_SIZE)
        
        # 3. 互斥重叠惩罚 (防止多根线挤在同一个像素上)
        flat_strokes = stroke_imgs_tensor.view(NUM_STROKES, -1) 
        overlaps = torch.matmul(flat_strokes, flat_strokes.T)   
        areas = torch.sum(flat_strokes, dim=1) + 1e-5
        ratios = overlaps / areas.unsqueeze(1)
        mask = ~torch.eye(NUM_STROKES, dtype=torch.bool, device=DEVICE)
        
        # 🌟 修复1：归一化！除以线条对的数量，防止数值爆炸
        num_pairs = NUM_STROKES * (NUM_STROKES - 1)
        base_overlap = ratios[mask].sum() / num_pairs
        
        # 🌟 修复2：退火算法 (Annealing)
        if epoch < 400:
            lambda_overlap = 0.0
        elif epoch < 800:
            # 在 400~800 轮之间，缓慢地、线性地把权重从 0 增加到 25.0
            lambda_overlap = 25.0 * ((epoch - 400) / 400.0) 
        else:
            lambda_overlap = 25.0
            
        loss_overlap = base_overlap * lambda_overlap
        
        # 4. Sparsity Loss (裁撤冗余笔画，同样使用平滑退火)
        if epoch < 800:
            lambda_sparse = 0.0
        elif epoch < 1200:
            lambda_sparse = 0.08 * ((epoch - 800) / 400.0)
        else:
            lambda_sparse = 0.08
            
        loss_sparse = lambda_sparse * total_alpha
        
        # ==========================================================

        loss = loss_mse * 20.0 + loss_iou * 5.0 + loss_gravity * 10.0 + loss_overlap + loss_sparse
        
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            # 打印时我们看看原始的 base_overlap 是多少
            print(f"Epoch {epoch:04d} | IoU: {loss_iou.item():.3f} | Overlap_Loss: {loss_overlap.item():.3f} (Weight: {lambda_overlap:.1f}) | Sparsity: {loss_sparse.item():.3f}")

    # ================== 可视化 ==================
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(target_mask.cpu().detach(), cmap='gray')
    plt.title("Target Glyph")
    
    plt.subplot(1, 3, 2)
    plt.imshow(pred_img.cpu().detach(), cmap='gray')
    plt.title("Reconstructed Tensors")
    
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_xlim(0, CANVAS_SIZE)
    ax3.set_ylim(CANVAS_SIZE, 0)
    ax3.set_aspect('equal')
    ax3.set_title("Anti-Collision Skeleton Paths")
    
    t_plot = torch.linspace(0, 1, 100, device=DEVICE).unsqueeze(1)
    for s in strokes:
        if torch.sigmoid(s.alpha_logit) > 0.5:
            m_pts = s.cubic_bezier(s.P, t_plot).detach().cpu().numpy()
            ax3.plot(m_pts[:, 0], m_pts[:, 1], linewidth=2.5, marker='.', markersize=1)

    plt.show()

if __name__ == "__main__":
    train_pure_pytorch()