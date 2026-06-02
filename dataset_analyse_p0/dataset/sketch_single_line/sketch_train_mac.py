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
from scipy.ndimage import distance_transform_edt # 引入距离变换工具

TARGET_DIR = "../alien_tensors_raw"
CANVAS_SIZE = 256
NUM_STROKES = 12
NUM_SAMPLES = 50

# 自动检测 MPS / CUDA
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ... (保留原有的 get_random_glyph_target 函数) ...
def get_random_glyph_target(font_dir, size=CANVAS_SIZE):
    font_files = glob.glob(os.path.join(font_dir, "*.[ot]tf"))
    if not font_files: raise ValueError("找不到字体文件！")
    
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
                draw.text(((size-(bbox[2]-bbox[0]))/2, (size-(bbox[3]-bbox[1]))/2), 
                          char, font=pil_font, fill=0)
                
                target = torch.from_numpy(np.array(img)).float() / 255.0
                target = target.unsqueeze(-1).repeat(1, 1, 4)
                target[:, :, 3] = 1.0 
                return target.to(DEVICE), font_path, char
        except: continue
    raise ValueError("未找到可渲染字符")


class GaussianStroke(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化在画布中心附近，防止一开始就被引力场撕碎
        self.P = nn.Parameter(torch.rand(4, 2) * (CANVAS_SIZE * 0.4) + (CANVAS_SIZE * 0.3))
        self.Q = nn.Parameter(torch.rand(4, 2) * 2.0)
        # 初始化 Alpha 为较高的值，鼓励模型先去覆盖，而不是一上来就摆烂消失
        self.alpha_logit = nn.Parameter(torch.tensor(2.0)) 

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
        img = torch.max(torch.exp(-dist_sq / (radii.unsqueeze(0)**2 + 1e-3)), dim=-1)[0]
        
        return img * torch.sigmoid(self.alpha_logit)

def train_pure_pytorch():
    target_img, _, char = get_random_glyph_target(TARGET_DIR)
    target_mask = (1.0 - target_img[:,:,0]).to(DEVICE)
    
    # 🌟 核心杀器 1：构建引力场 (Distance Transform)
    # 背景为 1，字体为 0。算出每个背景像素距离字体的距离
    bg_mask_np = (target_img[:,:,0] > 0.5).cpu().numpy() 
    dt_map_np = distance_transform_edt(bg_mask_np)
    # 归一化并传入计算设备
    dt_map_np = dt_map_np / (np.max(dt_map_np) + 1e-6)
    gravity_field = torch.tensor(dt_map_np, dtype=torch.float32, device=DEVICE)

    x = torch.linspace(0, CANVAS_SIZE, CANVAS_SIZE, device=DEVICE)
    y = torch.linspace(0, CANVAS_SIZE, CANVAS_SIZE, device=DEVICE)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

    strokes = nn.ModuleList([GaussianStroke().to(DEVICE) for _ in range(NUM_STROKES)])
    
    # 采用带有动量的 Adam，并且学习率适当降低以防震荡
    optimizer = optim.Adam(strokes.parameters(), lr=1.0)
    
    print(f"🎯 训练字符: {char} | 设备: {DEVICE}")

    for epoch in range(1200):
        optimizer.zero_grad()
        
        pred_img = torch.zeros(CANVAS_SIZE, CANVAS_SIZE, device=DEVICE)
        total_alpha = 0
        for s in strokes:
            pred_img += s(grid_x, grid_y)
            total_alpha += torch.sigmoid(s.alpha_logit)
            
        pred_img = torch.clamp(pred_img, 0, 1)
        
        # ==================== 🌟 终极 Loss 组合 ====================
        
        # 1. Soft-IoU Loss (彻底解决面积覆盖和超出问题)
        intersection = torch.sum(pred_img * target_mask)
        union = torch.sum(pred_img) + torch.sum(target_mask) - intersection
        loss_iou = 1.0 - (intersection + 1e-5) / (union + 1e-5)
        
        # 2. Gravity Loss (惩罚出界：把乱跑的控制点强行吸回来！)
        # 预测图超出的部分乘以引力场，偏离越远惩罚呈指数级增加
        loss_gravity = torch.sum(pred_img * gravity_field) / (CANVAS_SIZE * CANVAS_SIZE)
        
        # 3. Sparsity Loss (渐进式稀疏，前期专注覆盖，后期裁撤冗余笔画)
        lambda_sparse = 0.001 if epoch < 600 else 0.05
        loss_sparse = lambda_sparse * total_alpha
        
        loss = loss_iou * 10.0 + loss_gravity * 50.0 + loss_sparse
        
        # ==========================================================

        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d} | Total Loss: {loss.item():.4f} | IoU Loss: {loss_iou.item():.4f}")

    # 训练结束后进行最终可视化
    plt.figure(figsize=(15, 5))
    
    # 图 1：目标字体
    plt.subplot(1, 3, 1)
    plt.imshow(target_mask.cpu().detach(), cmap='gray')
    plt.title("Target Glyph")
    
    # 图 2：重建的图像
    plt.subplot(1, 3, 2)
    plt.imshow(pred_img.cpu().detach(), cmap='gray')
    plt.title("Reconstructed Tensors")
    
    # 图 3：纯净骨架与控制点展示
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_xlim(0, CANVAS_SIZE)
    ax3.set_ylim(CANVAS_SIZE, 0) # Y轴翻转，与图像坐标系一致
    ax3.set_aspect('equal')
    ax3.set_title("Extracted Skeleton Paths")
    
    t_plot = torch.linspace(0, 1, 100, device=DEVICE).unsqueeze(1)
    for s in strokes:
        # 只画出存在概率 > 0.5 的线条，证明稀疏 Loss 起效了
        if torch.sigmoid(s.alpha_logit) > 0.5:
            m_pts = s.cubic_bezier(s.P, t_plot).detach().cpu().numpy()
            ax3.plot(m_pts[:, 0], m_pts[:, 1], linewidth=2, marker='.', markersize=2)

    plt.show()

if __name__ == "__main__":
    train_pure_pytorch()