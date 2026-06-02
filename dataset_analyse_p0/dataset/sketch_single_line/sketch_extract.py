import os
import random
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from skimage import morphology
from skan import Skeleton, summarize
from scipy.optimize import least_squares
import cv2
import json
import matplotlib.cm as cm

# ==========================================
# ⚙️ 硬件与全局配置
# ==========================================
TARGET_DIR = "../alien_tensors_raw"  # 替换为你的字体目录
CANVAS_SIZE = 256

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"🚀 渲染设备: {DEVICE} | 模式: 显式几何管线 (瞬间完成, 无需Epoch训练)")

# ==========================================
# 🎯 0. 数据生成器
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
                # 转换为张量 [H, W, 4]
                target = torch.from_numpy(np.array(img)).float() / 255.0
                return target.unsqueeze(-1).repeat(1, 1, 4).to(DEVICE), font_path, char
        except: continue
    raise ValueError("未找到可渲染字符")

# ==========================================
# 🧠 1 & 2. 轮廓与骨架提取
# ==========================================
def extract_skeleton_and_contour(binary_img):
    # 提取骨架 (单像素宽)
    skeleton = morphology.skeletonize(binary_img)
    
    # 提取轮廓 (用于后续宽度计算)
    contours, _ = cv2.findContours((binary_img*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise ValueError("未检测到轮廓")
    # 安全处理点集形状
    contour_pts = np.vstack(contours).reshape(-1, 2)
    return skeleton, contour_pts

# ==========================================
# 🧠 3. 骨架 -> 拓扑图 (Graph)
# ==========================================
def build_skeleton_graph(skeleton_img):
    skel_obj = Skeleton(skeleton_img)
    branch_data = summarize(skel_obj)
    
    edges = []
    # 遍历骨架图中的每一条线段 (Branch)
    for index, row in branch_data.iterrows():
        # 获取这条边上的所有像素点坐标序列 (y, x) -> (x, y)
        coords = skel_obj.path_coordinates(index)
        path_xy = np.column_stack([coords[:, 1], coords[:, 0]])
        
        node_start = row['node-id-src']
        node_end = row['node-id-dst']
        
        edges.append({
            'start_node': node_start,
            'end_node': node_end,
            'path_pixels': path_xy
        })
    return edges

# ==========================================
# 🧠 4. 离散像素 -> 贝塞尔控制点拟合
# ==========================================
def cubic_bezier_np(P, t):
    mt = 1 - t
    return mt**3 * P[0] + 3*mt**2*t * P[1] + 3*mt*t**2 * P[2] + t**3 * P[3]

def fit_bezier_to_pixels(pixel_path):
    num_pts = len(pixel_path)
    if num_pts < 2: return None
    
    # 强制锁定起点和终点 (解决端点乱飞问题)
    P0 = pixel_path[0]
    P3 = pixel_path[-1]
    t_vals = np.linspace(0, 1, num_pts)[:, np.newaxis]
    
    P1_init = P0 + (P3 - P0) * 0.33
    P2_init = P0 + (P3 - P0) * 0.66
    
    def residuals(controls):
        P1 = controls[0:2]
        P2 = controls[2:4]
        P_all = np.array([P0, P1, P2, P3])
        curve_pts = cubic_bezier_np(P_all, t_vals)
        return (curve_pts - pixel_path).flatten()

    res = least_squares(residuals, x0=np.concatenate([P1_init, P2_init]))
    return np.array([P0, res.x[0:2], res.x[2:4], P3])

# ==========================================
# 🧠 5. 宽度回归 (1D Bezier)
# ==========================================
def regress_width(mother_bezier, contour_pts):
    t_vals = np.linspace(0, 1, 50)[:, np.newaxis]
    m_pts = cubic_bezier_np(mother_bezier, t_vals)
    
    # 计算母线上 50 个点到字体轮廓的最短距离 (即局部真实半径)
    from scipy.spatial.distance import cdist
    dists = cdist(m_pts, contour_pts)
    local_radii = np.min(dists, axis=1) 
    
    def width_residuals(W):
        mt = 1 - t_vals.flatten()
        t = t_vals.flatten()
        W_curve = mt**3*W[0] + 3*mt**2*t*W[1] + 3*mt*t**2*W[2] + t**3*W[3]
        return W_curve - local_radii
        
    res = least_squares(width_residuals, x0=np.array([4.0, 4.0, 4.0, 4.0]))
    # 绝对值保护，防止出现负宽度
    return np.abs(res.x)

# ==========================================
# 🚀 主管线执行
# ==========================================
def run_geometry_pipeline():
    # 1. 获取目标图像
    target_img, _, char = get_random_glyph_target(TARGET_DIR)
    
    # 将 PyTorch 图像转换为 NumPy 二值化数组 (字=1, 底=0)
    target_mask = (1.0 - target_img[:,:,0]).cpu().numpy()
    binary_img = target_mask > 0.5
    
    print(f"🎯 正在解析字符: '{char}'...")

    # 2. 提取像素级骨架与轮廓
    skeleton, contour_pts = extract_skeleton_and_contour(binary_img)
    
    # 3. 构建拓扑图
    edges = build_skeleton_graph(skeleton)
    print(f"🔗 Skan 成功解析出 {len(edges)} 条拓扑分支")

    final_tokens = []
    
    # 4 & 5. 遍历每一条分支，进行 Bezier 拟合与宽度回归
    for idx, edge in enumerate(edges):
        path_pixels = edge['path_pixels']
        if len(path_pixels) < 4: continue # 过滤掉极短的噪点分支
        
        # 拟合母线
        P_opt = fit_bezier_to_pixels(path_pixels)
        if P_opt is None: continue
        
        # 拟合宽度
        W_opt = regress_width(P_opt, contour_pts)
        
        final_tokens.append({
            "stroke_id": idx,
            "mother_bezier": P_opt.tolist(),
            "width_bezier_1d": W_opt.tolist()
        })

    print(f"✅ 提取完成！生成了 {len(final_tokens)} 个完美的 1D 张量 Token。")
    
    # ==========================================
    # 📊 终极 1x2 预览渲染 (沿用你最爱的 PyTorch 高斯渲染)
    # ==========================================
    # 注意：这里的网格必须是 y 在前，x 在后，对应图像的行和列
# 注意：这里的网格必须是 y 在前，x 在后，对应图像的行和列
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, CANVAS_SIZE - 1, CANVAS_SIZE, device=DEVICE),
        torch.linspace(0, CANVAS_SIZE - 1, CANVAS_SIZE, device=DEVICE), 
        indexing='ij'
    )
    
    # 创建一个纯白色的 RGB 画布 [256, 256, 3]
    clean_pred_img = torch.ones(CANVAS_SIZE, CANVAS_SIZE, 3, device=DEVICE)
    t_plot = torch.linspace(0, 1, 200, device=DEVICE).unsqueeze(1) 
    
    # 防止因提取失败导致 len 为 0 报错
    num_tokens = max(1, len(final_tokens))
    cmap = cm.get_cmap('tab20', num_tokens)

    for idx, token in enumerate(final_tokens):
        P_tensor = torch.tensor(token["mother_bezier"], dtype=torch.float32, device=DEVICE)
        W_tensor = torch.tensor(token["width_bezier_1d"], dtype=torch.float32, device=DEVICE)
        
        # 取出对应的 RGB 颜色 
        color = torch.tensor(cmap(idx)[:3], dtype=torch.float32, device=DEVICE)
        
        mt = 1 - t_plot
        M_t = mt**3*P_tensor[0] + 3*mt**2*t_plot*P_tensor[1] + 3*mt*t_plot**2*P_tensor[2] + t_plot**3*P_tensor[3]
        radii = torch.abs(mt**3*W_tensor[0] + 3*mt**2*t_plot*W_tensor[1] + 3*mt*t_plot**2*W_tensor[2] + t_plot**3*W_tensor[3]).squeeze()
        
        # 计算网格点到母线上每个采样点的欧氏距离
        dist = torch.sqrt((grid_x.unsqueeze(-1) - M_t[:, 0])**2 + (grid_y.unsqueeze(-1) - M_t[:, 1])**2)
        
        # 硬边缘抗锯齿渲染
        alpha_mask = torch.clamp(radii.view(1, 1, -1) - dist + 0.5, 0, 1) 
        stroke_alpha = torch.max(alpha_mask, dim=-1)[0].unsqueeze(-1)
        
        # 颜色合并 (Alpha Blending)
        clean_pred_img = clean_pred_img * (1 - stroke_alpha) + color * stroke_alpha

    clean_pred_img_np = clean_pred_img.cpu().numpy()

    # 🌟 核心修复：加大画布 (14,7)，提升显示器 DPI (120)，改用 subplots 精确控制
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=120)
    
    # 左图：原始字符
    axes[0].imshow(target_img[:,:,0].cpu().numpy(), cmap='gray')
    # pad=20 给标题留出足够空间，防止往下挤压图片
    axes[0].set_title(f"Target Glyph: '{char}'", fontsize=16, pad=20)
    axes[0].axis("off")

    # 右图：锐利且彩色的 1D 重建图
    axes[1].imshow(clean_pred_img_np)
    axes[1].set_title(f"Vectorized & Colored ({len(final_tokens)} Strokes)", fontsize=16, pad=20)
    axes[1].axis("off")

    # 🌟 增加布局的内边距，确保下半部分绝对不会被裁掉
    plt.tight_layout(pad=3.0)
    plt.show()

if __name__ == "__main__":
    run_geometry_pipeline()