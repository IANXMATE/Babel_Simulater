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
import json

TARGET_DIR = "../alien_tensors_raw"
CANVAS_SIZE = 256
NUM_STROKES = 15      # 稍微增加笔画数，因为贴边缘初始化后，单根线覆盖面积会变小
NUM_SAMPLES = 60

# 自动检测 MPS / CUDA
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
    def __init__(self, contour_points):
        super().__init__()
        # 🌟 核心修改 1：局部轮廓追踪初始化！
        num_pts = contour_points.shape[0]
        
        # 1. 在轮廓上随机选一个起点
        start_idx = torch.randint(0, num_pts, (1,), device=DEVICE).item()
        p0 = contour_points[start_idx]
        
        # 2. 计算轮廓上所有点到起点的欧氏距离
        dists = torch.norm(contour_points - p0, dim=1)
        
        # 3. 按距离排序，获取起点的“局部邻居”
        sorted_indices = torch.argsort(dists)
        
        # 4. 按步长取 4 个点。步长决定了初始曲线的长度 (比如取第 0, 10, 20, 30 近的点)
        step = max(1, num_pts // 80) 
        # 确保索引不越界
        idx0 = sorted_indices[0]
        idx1 = sorted_indices[min(step, num_pts-1)]
        idx2 = sorted_indices[min(step*2, num_pts-1)]
        idx3 = sorted_indices[min(step*3, num_pts-1)]
        
        idx = torch.stack([idx0, idx1, idx2, idx3])
        self.P = nn.Parameter(contour_points[idx].clone()) 
        
        # 宽度曲线：因为母线已经在边缘了，初始化宽度给稍微大一点点，让它向内侧探索
        self.Q = nn.Parameter(torch.tensor([[0.0, 0.0], [5.0, 8.0], [15.0, 8.0], [20.0, 0.0]], device=DEVICE))
        
        # 存在概率：初始值极高
        self.alpha_logit = nn.Parameter(torch.tensor(5.0, device=DEVICE)) 

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
        img_sum = torch.sum(torch.exp(-dist_sq / (radii.unsqueeze(0)**2 + 1e-2)), dim=-1)
        img = torch.clamp(img_sum, 0, 1) 
        
        return img * torch.sigmoid(self.alpha_logit)


def train_pure_pytorch():
    target_img, _, char = get_random_glyph_target(TARGET_DIR)
    target_mask = (1.0 - target_img[:,:,0]).to(DEVICE)
    
    # 🌟 核心修改 2：利用形态学提取 TTF 图像的单像素边界轮廓
    # 膨胀 - 腐蚀 = 边缘
    tm_unsqueeze = target_mask.unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(tm_unsqueeze, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-tm_unsqueeze, kernel_size=3, stride=1, padding=1)
    contour_mask = (dilated - eroded).squeeze()
    
    # 获取所有的轮廓像素坐标
    y_idx, x_idx = torch.where(contour_mask > 0.5)
    contour_points = torch.stack([x_idx, y_idx], dim=1).float()
    
    bg_mask_np = (target_img[:,:,0] > 0.5).cpu().numpy() 
    dt_map_np = distance_transform_edt(bg_mask_np)
    dt_map_np = dt_map_np / (np.max(dt_map_np) + 1e-6)
    gravity_field = torch.tensor(dt_map_np, dtype=torch.float32, device=DEVICE)

    x = torch.linspace(0, CANVAS_SIZE, CANVAS_SIZE, device=DEVICE)
    y = torch.linspace(0, CANVAS_SIZE, CANVAS_SIZE, device=DEVICE)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

    strokes = nn.ModuleList([GaussianStroke(contour_points).to(DEVICE) for _ in range(NUM_STROKES)])
    optimizer = optim.Adam(strokes.parameters(), lr=0.5)
    
    print(f"🎯 训练字符: {char} | 设备: {DEVICE} | 轮廓像素总数: {contour_points.shape[0]}")

    for epoch in range(1500):
        optimizer.zero_grad()
        
        stroke_imgs = []
        total_alpha = 0
        for s in strokes:
            if epoch < 300:
                s.alpha_logit.requires_grad = False
            else:
                s.alpha_logit.requires_grad = True
                
            img = s(grid_x, grid_y)
            stroke_imgs.append(img)
            total_alpha += torch.sigmoid(s.alpha_logit)
            
        stroke_imgs_tensor = torch.stack(stroke_imgs)
        pred_img = torch.clamp(torch.sum(stroke_imgs_tensor, dim=0), 0, 1)
        
        # 1. Soft-IoU & MSE
        intersection = torch.sum(pred_img * target_mask)
        union = torch.sum(pred_img) + torch.sum(target_mask) - intersection
        loss_iou = 1.0 - (intersection + 1e-5) / (union + 1e-5)
        loss_mse = F.mse_loss(pred_img, target_mask)
        
        # 2. Gravity
        loss_gravity = torch.sum(pred_img * gravity_field) / (CANVAS_SIZE * CANVAS_SIZE)
        
        # 3. Overlap (归一化 + 退火)
        flat_strokes = stroke_imgs_tensor.view(NUM_STROKES, -1) 
        overlaps = torch.matmul(flat_strokes, flat_strokes.T)   
        areas = torch.sum(flat_strokes, dim=1) + 1e-5
        ratios = overlaps / areas.unsqueeze(1)
        mask = ~torch.eye(NUM_STROKES, dtype=torch.bool, device=DEVICE)
        
        num_pairs = NUM_STROKES * (NUM_STROKES - 1)
        base_overlap = ratios[mask].sum() / num_pairs
        
        if epoch < 400:
            lambda_overlap = 0.0
        elif epoch < 800:
            lambda_overlap = 25.0 * ((epoch - 400) / 400.0) 
        else:
            lambda_overlap = 25.0
            
        loss_overlap = base_overlap * lambda_overlap
        
        # 4. Sparsity (退火)
        if epoch < 800:
            lambda_sparse = 0.0
        elif epoch < 1200:
            lambda_sparse = 0.08 * ((epoch - 800) / 400.0)
        else:
            lambda_sparse = 0.08
            
        loss_sparse = lambda_sparse * total_alpha
        
        loss = loss_mse * 20.0 + loss_iou * 5.0 + loss_gravity * 10.0 + loss_overlap + loss_sparse
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d} | IoU: {loss_iou.item():.3f} | Overlap: {loss_overlap.item():.3f} | Sparsity: {loss_sparse.item():.3f}")

    # ================== 🧹 NMS 清洗与最终展示 ==================
    print("\n🚀 执行 NMS 提取...")
    ALPHA_THRESHOLD = 0.85  
    candidate_strokes = []

    for i, s in enumerate(strokes):
        alpha_prob = torch.sigmoid(s.alpha_logit).item()
        if alpha_prob > ALPHA_THRESHOLD:
            candidate_strokes.append({
                "stroke_id": i,
                "confidence": alpha_prob,
                "P": s.P.detach().cpu(), 
                "Q": s.Q.detach().cpu()  
            })

    candidate_strokes = sorted(candidate_strokes, key=lambda x: x["confidence"], reverse=True)
    final_tensors = []
    NMS_DISTANCE_THRESHOLD = 15.0 

    for candidate in candidate_strokes:
        is_redundant = False
        for kept_stroke in final_tensors:
            dist = torch.mean(torch.norm(candidate["P"] - kept_stroke["P"], dim=-1)).item()
            if dist < NMS_DISTANCE_THRESHOLD:
                is_redundant = True
                break
        if not is_redundant:
            final_tensors.append(candidate)

    export_data = []
    for item in final_tensors:
        export_data.append({
            "stroke_id": item["stroke_id"],
            "confidence": round(item["confidence"], 4),
            "mother_bezier": item["P"].numpy().tolist(),
            "width_bezier": item["Q"].numpy().tolist()
        })

    print(f"✅ 清洗完成！保留 {len(export_data)} 根。")
    # with open("alien_skeleton_tensors.json", "w") as f:
    #     json.dump(export_data, f, indent=4)

    # 可视化
    clean_pred_img = torch.zeros(CANVAS_SIZE, CANVAS_SIZE, device=DEVICE)
    t_plot = torch.linspace(0, 1, 100, device=DEVICE).unsqueeze(1)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    
    # 画出原始轮廓线以作对比 (用彩色标出轮廓)
    target_disp = target_mask.cpu().detach().numpy().copy()
    contour_disp = contour_mask.cpu().detach().numpy()
    target_disp[contour_disp > 0.5] = 0.5 # 轮廓处标灰
    
    plt.imshow(target_disp, cmap='gray')
    plt.title("Target & Contours (Grey)")

    ax3 = plt.subplot(1, 3, 3)
    ax3.set_xlim(0, CANVAS_SIZE); ax3.set_ylim(CANVAS_SIZE, 0); ax3.set_aspect('equal')
    ax3.set_title(f"Final Skeleton (NMS Filtered: {len(export_data)})")

    for item in final_tensors:
        P_tensor = item["P"].to(DEVICE)
        Q_tensor = item["Q"].to(DEVICE)
        
        mt = 1 - t_plot
        M_t = mt**3*P_tensor[0] + 3*mt**2*t_plot*P_tensor[1] + 3*mt*t_plot**2*P_tensor[2] + t_plot**3*P_tensor[3]
        W_t = mt**3*Q_tensor[0] + 3*mt**2*t_plot*Q_tensor[1] + 3*mt*t_plot**2*Q_tensor[2] + t_plot**3*Q_tensor[3]
        base = P_tensor[3] - P_tensor[0]
        radii = torch.abs(torch.matmul(W_t - Q_tensor[0], torch.stack([-base[1], base[0]]) / (torch.norm(base)+1e-6)))
        
        dist_sq = (grid_x.unsqueeze(-1) - M_t[:, 0])**2 + (grid_y.unsqueeze(-1) - M_t[:, 1])**2
        img = torch.max(torch.exp(-dist_sq / (radii.unsqueeze(0)**2 + 1e-2)), dim=-1)[0]
        clean_pred_img += img
        
        m_pts = M_t.detach().cpu().numpy()
        ax3.plot(m_pts[:, 0], m_pts[:, 1], linewidth=3.0, marker='.', markersize=2)

    clean_pred_img = torch.clamp(clean_pred_img, 0, 1)
    plt.subplot(1, 3, 2)
    plt.imshow(clean_pred_img.cpu().detach(), cmap='gray')
    plt.title("Clean Reconstructed")

    plt.show()

if __name__ == "__main__":
    train_pure_pytorch()