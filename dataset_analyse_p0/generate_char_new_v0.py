import torch
import random
import math
import numpy as np
from PIL import Image

def calculate_bounding_box(sequence):
    """
    空跑数学引擎，精准探测物理极值，防止缩成一个点
    """
    pts_x, pts_y = [], []
    prev_x, prev_y = 0.0, 0.0
    
    for token in sequence:
        if isinstance(token, torch.Tensor):
            token = token.tolist()
        dx, dy, L, theta, kappa, W, P = token
        if P == 1.0: break
            
        start_x = prev_x + dx
        start_y = prev_y + dy
        
        steps = 20 
        t_vals = np.linspace(0, 1.0, steps)
        for t in t_vals:
            if abs(kappa) < 1e-4:
                math_x = start_x + L * t * math.cos(theta)
                math_y = start_y + L * t * math.sin(theta)
            else:
                math_x = start_x + (L / kappa) * (math.sin(theta + kappa * t) - math.sin(theta))
                math_y = start_y - (L / kappa) * (math.cos(theta + kappa * t) - math.cos(theta))
            
            pts_x.append(math_x)
            pts_y.append(math_y)
            
        prev_x, prev_y = math_x, math_y

    if not pts_x:
        return 0, 0, 0, 0

    return min(pts_x), max(pts_x), min(pts_y), max(pts_y)


def render_alien_glyph(sequence, canvas_size=512):
    """
    异界符文渲染：黑底白字、锐化边缘、神经质粗细变化
    """
    min_x, max_x, min_y, max_y = calculate_bounding_box(sequence)
    math_width = max(max_x - min_x, 1e-5)
    math_height = max(max_y - min_y, 1e-5)
    
    target_size = canvas_size * 0.8
    scale = target_size / max(math_width, math_height)
    
    math_center_x = (min_x + max_x) / 2.0
    math_center_y = (min_y + max_y) / 2.0
    
    cx = canvas_size / 2.0 - math_center_x * scale
    cy = canvas_size / 2.0 - math_center_y * scale

    density_map = np.zeros((canvas_size, canvas_size), dtype=np.float32)
    prev_x, prev_y = 0.0, 0.0
    
    for token in sequence:
        if isinstance(token, torch.Tensor):
            token = token.tolist()
        dx, dy, L, theta, kappa, W, P = token
        if P == 1.0: break
            
        start_x = prev_x + dx
        start_y = prev_y + dy
        
        steps = int(max(L * scale * 0.5, 50)) 
        t_vals = np.linspace(0, 1.0, steps)
        
        for i, t in enumerate(t_vals):
            if abs(kappa) < 1e-4:
                math_x = start_x + L * t * math.cos(theta)
                math_y = start_y + L * t * math.sin(theta)
            else:
                math_x = start_x + (L / kappa) * (math.sin(theta + kappa * t) - math.sin(theta))
                math_y = start_y - (L / kappa) * (math.cos(theta + kappa * t) - math.cos(theta))
            
            px = int(cx + math_x * scale)
            py = int(cy + math_y * scale) 
            
            # ==========================================
            # 👽 核心修复区：物理与像素强制解耦 
            # ==========================================
            progress = t
            neurotic_jitter = math.sin(progress * math.pi * 4) * 0.3 
            kappa_mutator = min(abs(kappa) * 0.5, 2.0)
            
            # 基础粗细不再乘以 scale，直接锚定为绝对像素值 (例如 2~8 像素)
            base_radius = max(W * 1.5, 2.0) 
            
            # 叠加神经质突变，并加上终极防爆锁 (绝不超过画布的 3%)
            sigma = base_radius * (1.0 + neurotic_jitter) * (1.0 + kappa_mutator)
            sigma = min(sigma, canvas_size * 0.03) 
            
            ink_alpha = 1.2 
            r = int(math.ceil(2.0 * sigma))
            
            y_min, y_max = max(0, py - r), min(canvas_size, py + r + 1)
            x_min, x_max = max(0, px - r), min(canvas_size, px + r + 1)
            
            if y_min >= y_max or x_min >= x_max:
                continue
                
            YY, XX = np.ogrid[y_min:y_max, x_min:x_max]
            dist_sq = (XX - px)**2 + (YY - py)**2
            
            gaussian_kernel = ink_alpha * np.exp(-dist_sq / (sigma**2 + 1e-3))
            density_map[y_min:y_max, x_min:x_max] += gaussian_kernel
            
        prev_x, prev_y = math_x, math_y

    # ==========================================
    # 赛博渲染：高对比度截断 + 黑底发光
    # ==========================================
    ink_intensity = np.clip(density_map, 0, 1.0)
    ink_intensity = np.power(ink_intensity, 0.5) 
    
    # 强硬的边缘截断，保留锐利的符文刻痕感
    ink_intensity[ink_intensity < 0.25] = 0.0
    
    # 【黑底白字】视觉翻转：浓度越高的地方越亮 (255=白)
    pixel_values = 255 * ink_intensity
    img_array = pixel_values.astype(np.uint8)
    
    return Image.fromarray(img_array, mode='L')


def preview_parametric_character(data_path="omniglot_parametric_7d.pt"):
    try:
        data = torch.load(data_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {data_path}，请确认路径。")
        return

    if not data:
        print("❌ 数据集是空的！")
        return

    sample = random.choice(data)
    label = sample["label"]
    seq = sample["sequence"]  

    print(f"🎲 抽取异界符文成功！ (Label: {label})")
    
    rendered_image = render_alien_glyph(sequence=seq, canvas_size=512)

    rendered_image.show(title=f"Alien_Glyph_{label}")
    
    # save_filename = f"cyber_rune_label_{label}.png"
    # rendered_image.save(save_filename)
    # print(f"✅ 赛博符文已具象化并保存至: {save_filename}")

if __name__ == "__main__":
    preview_parametric_character()