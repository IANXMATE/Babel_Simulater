import os
import math
import torch
import numpy as np

# ==========================================
# 1. 核心数学拟合引擎 (Curve Fitting)
# ==========================================
def normalize_angle(angle):
    """将角度强制映射到 [-pi, pi]"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def extract_stroke_parameters(stroke_points, prev_end_x, prev_end_y):
    """
    【核心黑魔法】将一堆散点压缩为 1 个高阶 7 维 Token
    返回: [dx, dy, L, theta, kappa, W, P]
    """
    if len(stroke_points) < 2:
        return None # 忽略无意义的点

    # 1. 提取起点与终点
    start_x, start_y = stroke_points[0]
    end_x, end_y = stroke_points[-1]

    # 2. 计算悬空位移 (dx, dy)
    dx = start_x - prev_end_x
    dy = start_y - prev_end_y

    # 3. 计算弧长 (Length: L)
    # 通过累加相邻两点的欧氏距离
    length = 0.0
    angles = [] # 记录每一小段的方向，为计算曲率做准备
    
    for i in range(1, len(stroke_points)):
        x1, y1 = stroke_points[i-1]
        x2, y2 = stroke_points[i]
        seg_dx = x2 - x1
        seg_dy = y2 - y1
        dist = math.hypot(seg_dx, seg_dy)
        
        if dist > 1e-4: # 忽略原地抖动
            length += dist
            angles.append(math.atan2(seg_dy, seg_dx))

    if length < 1.0 or not angles:
        return None # 长度太短的杂乱笔画直接丢弃

    # 4. 计算初始射出角 (Theta: θ)
    # 取前一段有效位移的角度，避免手抖
    theta = angles[0]

    # 5. 计算曲率 (Curvature: κ)
    # 策略：累加所有角度的变化量。如果是直线，变化量接近0；如果是圆，变化量接近 2π 或 -2π。
    curvature = 0.0
    for i in range(1, len(angles)):
        delta_angle = normalize_angle(angles[i] - angles[i-1])
        curvature += delta_angle

    # 6. 基础宽度 (Width: W)
    # 作为一个初始常数，后续可以让 Transformer 自己去学习生成带有压感的宽度
    width = 1.0 

    # 7. 状态机 (State: P)
    # 0 代表正常笔画，后续会添加专门的 EOS Token
    p_state = 0.0

    return [dx, dy, length, theta, curvature, width, p_state], (end_x, end_y)


# ==========================================
# 2. 复杂度过滤器 (沿用经典逻辑)
# ==========================================
def is_complex_enough(strokes, min_strokes=3):
    # 在参数化模型中，我们不再关心总点数，因为它们全会被压缩！
    # 我们只关心它由多少“笔”组成。
    return len(strokes) >= min_strokes

# ==========================================
# 3. 主处理流程
# ==========================================
def process_omniglot_parametric(input_dir, output_file, min_strokes=3):
    if not os.path.exists(input_dir):
        print(f"[致命错误] 找不到文件夹 '{input_dir}'")
        return

    all_sequences = []
    total_files = 0
    
    print(f"🚀 正在启动矢量拟合引擎，扫描 '{input_dir}'...")
    print("✨ 目标架构: 7维高阶参数化 Token [dx, dy, L, θ, κ, W, P]\n")

    for root, _, files in os.walk(input_dir):
        for filename in files:
            if not filename.endswith('.txt'):
                continue
                
            total_files += 1
            filepath = os.path.join(root, filename)
            
            strokes = []
            current_stroke = []
            
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line == "START":
                        continue
                    elif line == "BREAK":
                        if current_stroke:
                            strokes.append(current_stroke)
                            current_stroke = []
                    else:
                        parts = line.replace(',', ' ').split()
                        if len(parts) >= 2:
                            try:
                                current_stroke.append((float(parts[0]), float(parts[1])))
                            except ValueError:
                                continue
            
            if current_stroke:
                strokes.append(current_stroke)

            if not strokes or not is_complex_enough(strokes, min_strokes):
                continue

            # ==========================================
            # 🎯 执行点云到方程的压缩 (Point Cloud to Parametric)
            # ==========================================
            sequence = []
            prev_end_x, prev_end_y = strokes[0][0][0], strokes[0][0][1] # 假设以第一笔起点为基准

            for stroke in strokes:
                result = extract_stroke_parameters(stroke, prev_end_x, prev_end_y)
                if result:
                    token, (prev_end_x, prev_end_y) = result
                    sequence.append(token)

            if not sequence:
                continue

            # 添加结束符 EOS (P=1)
            sequence.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]) 
            
            seq_tensor = torch.tensor(sequence, dtype=torch.float32)
            
            path_parts = os.path.normpath(root).split(os.sep)
            label = f"{path_parts[-2]}_{path_parts[-1]}" if len(path_parts) >= 2 else "Unknown"

            all_sequences.append({
                "label": label,
                "filename": filename,
                "sequence": seq_tensor
            })

    # ==========================================
    # 4. 空间参数的防爆归一化
    # ==========================================
    kept_files = len(all_sequences)
    print("-" * 45)
    print(f"📊 扫描总计: {total_files} 个字符")
    print(f"✅ 成功提取方程: {kept_files} 个")
    print("-" * 45)

    if kept_files == 0:
        return

    # 🚨 极度关键：只对空间尺度 (dx, dy, L) 进行缩放，绝不能碰角度 θ 和曲率 κ！
    all_dx = torch.cat([item["sequence"][:-1, 0] for item in all_sequences])
    all_dy = torch.cat([item["sequence"][:-1, 1] for item in all_sequences])
    all_L = torch.cat([item["sequence"][:-1, 2] for item in all_sequences])
    
    spatial_data = torch.cat([all_dx, all_dy, all_L])
    global_scale = spatial_data.abs().max().item() + 1e-6
    
    print(f"计算得出全局空间缩放因子 (Max Absolute): {global_scale:.4f}")

    for item in all_sequences:
        seq = item["sequence"]
        # 对 dx, dy, Length 进行统一缩放，保持几何比例完美不变
        seq[:, 0] = seq[:, 0] / global_scale
        seq[:, 1] = seq[:, 1] / global_scale
        seq[:, 2] = seq[:, 2] / global_scale
        item["sequence"] = seq

    torch.save(all_sequences, output_file)
    print(f"🎉 参数化 7 维字库已完美保存至: {output_file}")
    
    # 展示恐怖的序列压缩率
    demo_seq = all_sequences[0]["sequence"]
    print(f"\n[降维打击展示] 序列长度暴降为: {len(demo_seq)} 步！(以前通常需要 200+ 步)")
    print(f"第一笔方程 [dx, dy, L, θ, κ, W, P]:\n  {demo_seq[0].tolist()}")

if __name__ == "__main__":
    INPUT_DIR = "strokes_background"  
    OUTPUT_FILE = "omniglot_parametric_7d.pt"
    
    process_omniglot_parametric(
        input_dir=INPUT_DIR, 
        output_file=OUTPUT_FILE, 
        min_strokes=3
    )