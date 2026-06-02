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

# ==========================================
# ⚙️ 硬件与全局配置
# ==========================================
TARGET_DIR = "../alien_tensors_raw"  # 替换为你的字体目录
CANVAS_SIZE = 256
NUM_STROKES = 15      # 笔画数量
NUM_SAMPLES = 60      # 贝塞尔曲线采样点数

# 自动检测 MPS (Apple Silicon) / CUDA
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"🚀 当前运行环境: {DEVICE}")

# ==========================================
# 🎯 数据生成器
# ==========================================
def get_random_glyph_target(font_dir, size=CANVAS_SIZE):
    font_files = glob.glob(os.path.join(font_dir, "*.[ot]tf"))
    if not font_files: raise ValueError("找不到字体文件")
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

# ==========================================
# 🧠 核心架构：1D 宽度标量的可微渲染器
# ==========================================
class GaussianStroke1D(nn.Module):
    def __init__(self, contour_points):
        super().__init__()
        # 1. 边缘轮廓追踪初始化 (母线 P)
        num_pts = contour_points.shape[0]
        start_idx = torch.randint(0, num_pts, (1,), device=DEVICE).item()
        p0 = contour_points[start_idx]
        dists = torch.norm(contour_points - p0, dim=1)
        sorted_indices = torch.argsort(dists)
        
        # 截取局部轮廓上的 4 个点
        step = max(1, num_pts // 80) 
        idx = torch.stack([
            sorted_indices[0],
            sorted_indices[min(step, num_pts-1)],
            sorted_indices[min(step*2, num_pts-1)],
            sorted_indices[min(step*3, num_pts-1)]
        ])
        self.P = nn.Parameter(contour_points[idx].clone()) 
        
        # 🌟 2. 核心大更新：使用 1D 标量来表示宽度 (4个纯粹的标量)
        # 初始化为绝对宽度 4.0 像素
        self.W = nn.Parameter(torch.tensor([4.0, 4.0, 4.0, 4.0], device=DEVICE))
        
        # 3. 存在概率
        self.alpha_logit = nn.Parameter(torch.tensor(5.0, device=DEVICE)) 

    # 二维坐标贝塞尔插值
    def cubic_bezier_2d(self, pts, t):
        mt = 1 - t
        return mt**3*pts[0] + 3*mt**2*t*pts[1] + 3*mt*t**2*pts[2] + t**3*pts[3]

    # 一维标量贝塞尔插值
    def cubic_bezier_1d(self, scalars, t):
        mt = 1 - t
        return mt**3*scalars[0] + 3*mt**2*t*scalars[1] + 3*mt*t**2*scalars[2] + t**3*scalars[3]

    def forward(self, grid_x, grid_y):
            t = torch.linspace(0, 1, NUM_SAMPLES, device=DEVICE).unsqueeze(1)
            
            # 获取母线轨迹
            M_t = self.cubic_bezier_2d(self.P, t)
            
            # 🌟 修复点 1：加 .squeeze() 把 [60, 1] 变成 [60]
            radii = torch.abs(self.cubic_bezier_1d(self.W, t)).squeeze()
            
            dist_sq = (grid_x.unsqueeze(-1) - M_t[:, 0])**2 + (grid_y.unsqueeze(-1) - M_t[:, 1])**2
            
            # 🌟 修复点 2：使用 radii.view(1, 1, -1) 完美对齐 [256, 256, 60]
            img_sum = torch.sum(torch.exp(-dist_sq / (radii.view(1, 1, -1)**2 + 1e-2)), dim=-1)
            img = torch.clamp(img_sum, 0, 1) 
            
            return img * torch.sigmoid(self.alpha_logit)

# ==========================================
# 🚀 主训练循环
# ==========================================
def train_pure_pytorch():
    target_img, _, char = get_random_glyph_target(TARGET_DIR)
    target_mask = (1.0 - target_img[:,:,0]).to(DEVICE)
    
    # 提取 TTF 边缘轮廓用于初始化
    tm_unsqueeze = target_mask.unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(tm_unsqueeze, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-tm_unsqueeze, kernel_size=3, stride=1, padding=1)
    contour_mask = (dilated - eroded).squeeze()
    
    y_idx, x_idx = torch.where(contour_mask > 0.5)
    contour_points = torch.stack([x_idx, y_idx], dim=1).float()
    
    # 构建出界引力场
    bg_mask_np = (target_img[:,:,0] > 0.5).cpu().numpy() 
    dt_map_np = distance_transform_edt(bg_mask_np)
    dt_map_np = dt_map_np / (np.max(dt_map_np) + 1e-6)
    gravity_field = torch.tensor(dt_map_np, dtype=torch.float32, device=DEVICE)

    x = torch.linspace(0, CANVAS_SIZE, CANVAS_SIZE, device=DEVICE)
    y = torch.linspace(0, CANVAS_SIZE, CANVAS_SIZE, device=DEVICE)
    grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')

    strokes = nn.ModuleList([GaussianStroke1D(contour_points).to(DEVICE) for _ in range(NUM_STROKES)])
    optimizer = optim.Adam(strokes.parameters(), lr=0.5)
    
    print(f"🎯 开始训练字符: '{char}' | 轮廓初始化锚点数: {contour_points.shape[0]}")

    for epoch in range(1500):
        optimizer.zero_grad()
        
        stroke_imgs = []
        total_alpha = 0
        for s in strokes:
            # 冻结前期 alpha，逼迫控制点优化
            if epoch < 300:
                s.alpha_logit.requires_grad = False
            else:
                s.alpha_logit.requires_grad = True
                
            img = s(grid_x, grid_y)
            stroke_imgs.append(img)
            total_alpha += torch.sigmoid(s.alpha_logit)
            
        stroke_imgs_tensor = torch.stack(stroke_imgs)
        pred_img = torch.clamp(torch.sum(stroke_imgs_tensor, dim=0), 0, 1)
        
        # 1. 拟合 Loss: Soft-IoU + MSE
        intersection = torch.sum(pred_img * target_mask)
        union = torch.sum(pred_img) + torch.sum(target_mask) - intersection
        loss_iou = 1.0 - (intersection + 1e-5) / (union + 1e-5)
        loss_mse = F.mse_loss(pred_img, target_mask)
        
        # 2. 引力场 Loss
        loss_gravity = torch.sum(pred_img * gravity_field) / (CANVAS_SIZE * CANVAS_SIZE)
        
        # 3. 互斥排斥 Loss (带退火)
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
        
        # 4. 稀疏裁撤 Loss (带退火)
        if epoch < 800:
            lambda_sparse = 0.0
        elif epoch < 1200:
            lambda_sparse = 0.08 * ((epoch - 800) / 400.0)
        else:
            lambda_sparse = 0.08
            
        loss_sparse = lambda_sparse * total_alpha
        
        # 综合计算与反传
        loss = loss_mse * 20.0 + loss_iou * 5.0 + loss_gravity * 10.0 + loss_overlap + loss_sparse
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d} | IoU: {loss_iou.item():.3f} | Overlap: {loss_overlap.item():.3f} | Sparsity: {loss_sparse.item():.3f}")

    # ================== 🧹 NMS 提取纯净 1D 张量 ==================
    print("\n🚀 正在执行 NMS (非极大值抑制) 提取纯净骨架...")
    ALPHA_THRESHOLD = 0.85  
    candidate_strokes = []

    for i, s in enumerate(strokes):
        alpha_prob = torch.sigmoid(s.alpha_logit).item()
        if alpha_prob > ALPHA_THRESHOLD:
            candidate_strokes.append({
                "stroke_id": i,
                "confidence": alpha_prob,
                "P": s.P.detach().cpu(), 
                "W": s.W.detach().cpu()   # 🌟 提取 1D W
            })

    # 按置信度排序
    candidate_strokes = sorted(candidate_strokes, key=lambda x: x["confidence"], reverse=True)
    final_tensors = []
    NMS_DISTANCE_THRESHOLD = 15.0 

    # 查重与过滤
    for candidate in candidate_strokes:
        is_redundant = False
        for kept_stroke in final_tensors:
            dist = torch.mean(torch.norm(candidate["P"] - kept_stroke["P"], dim=-1)).item()
            if dist < NMS_DISTANCE_THRESHOLD:
                is_redundant = True
                break
        if not is_redundant:
            final_tensors.append(candidate)

    # 序列化为最终的 Token 格式
    export_data = []
    for item in final_tensors:
        export_data.append({
            "stroke_id": item["stroke_id"],
            "confidence": round(item["confidence"], 4),
            "mother_bezier": item["P"].numpy().tolist(),     # [[x,y] x 4]
            "width_bezier_1d": item["W"].numpy().tolist()    # [w0, w1, w2, w3] 纯粹标量！
        })

    print(f"✅ 清洗完成！最终保留 {len(export_data)} 根有效 1D 骨架。")
    # with open("alien_skeleton_tensors_1d.json", "w") as f:
    #     json.dump(export_data, f, indent=4)

    # ================== 📊 纯净可视化重建 ==================
    clean_pred_img = torch.zeros(CANVAS_SIZE, CANVAS_SIZE, device=DEVICE)
    t_plot = torch.linspace(0, 1, 100, device=DEVICE).unsqueeze(1)

    plt.figure(figsize=(15, 5))
    
    # 视图 1：原始图像带轮廓标示
    plt.subplot(1, 3, 1)
    target_disp = target_mask.cpu().detach().numpy().copy()
    contour_disp = contour_mask.cpu().detach().numpy()
    target_disp[contour_disp > 0.5] = 0.5 
    plt.imshow(target_disp, cmap='gray')
    plt.title("Target Glyph & Contours")

    # 视图 3：提取的纯净骨架图
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_xlim(0, CANVAS_SIZE); ax3.set_ylim(CANVAS_SIZE, 0); ax3.set_aspect('equal')
    ax3.set_title(f"Extracted 1D Skeletons (Count: {len(export_data)})")

    for item in final_tensors:
            P_tensor = item["P"].to(DEVICE)
            W_tensor = item["W"].to(DEVICE)
            
            # 重新利用 1D 宽度进行渲染核对
            mt = 1 - t_plot
            M_t = mt**3*P_tensor[0] + 3*mt**2*t_plot*P_tensor[1] + 3*mt*t_plot**2*P_tensor[2] + t_plot**3*P_tensor[3]
            
            # 🌟 修复点 3：这里也加上 .squeeze()
            radii = torch.abs(mt**3*W_tensor[0] + 3*mt**2*t_plot*W_tensor[1] + 3*mt*t_plot**2*W_tensor[2] + t_plot**3*W_tensor[3]).squeeze()
            
            dist_sq = (grid_x.unsqueeze(-1) - M_t[:, 0])**2 + (grid_y.unsqueeze(-1) - M_t[:, 1])**2
            
            # 🌟 修复点 4：使用 radii.view(1, 1, -1)
            img = torch.max(torch.exp(-dist_sq / (radii.view(1, 1, -1)**2 + 1e-2)), dim=-1)[0]
            clean_pred_img += img
            
            m_pts = M_t.detach().cpu().numpy()
            ax3.plot(m_pts[:, 0], m_pts[:, 1], linewidth=3.0, marker='.', markersize=2)

    # 视图 2：NMS 过滤后的最终渲染成效
    clean_pred_img = torch.clamp(clean_pred_img, 0, 1)
    plt.subplot(1, 3, 2)
    plt.imshow(clean_pred_img.cpu().detach(), cmap='gray')
    plt.title("Clean 1D Reconstructed")

    plt.show()

if __name__ == "__main__":
    train_pure_pytorch()