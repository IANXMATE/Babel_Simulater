import torch
import random
import matplotlib.pyplot as plt

def preview_cartesian_dynamic_width(data_path="omniglot_cartesian_5d.pt"):
    try:
        data = torch.load(data_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {data_path}，请确认路径。")
        return

    if not data:
        print("❌ 数据集是空的！")
        return

    # ==========================================
    # 1. 随机抽签
    # ==========================================
    sample = random.choice(data)
    label = sample["label"]
    filename = sample["filename"]
    seq = sample["sequence"]  

    print(f"🎲 随机抽取成功！")
    print(f"🏷️ 语言/字符集: {label}")
    print(f"📄 原始文件名: {filename}")

    # ==========================================
    # 2. 坐标解析与速度记录 (核心改进)
    # ==========================================
    x, y = 0.0, 0.0
    strokes = [] 
    current_stroke = []
    all_speeds = [] # 记录这个字所有小步的移动速度
    
    for token in seq:
        dx, dy, p1, p2, p3 = token.tolist()
        
        if p3 == 1.0:
            if current_stroke:
                strokes.append(current_stroke)
            break
            
        prev_x, prev_y = x, y
        x += dx
        y += dy
        
        # 计算当前步的实际速度 (欧氏距离)
        speed = (dx**2 + dy**2)**0.5
        
        if p1 == 1.0:
            if not current_stroke:
                # 记录起点坐标，且起点的初始速度设为 0
                current_stroke.append((prev_x, prev_y, 0.0)) 
            # 记录坐标以及到达该坐标的速度
            current_stroke.append((x, y, speed))
            all_speeds.append(speed)
            
        elif p2 == 1.0:
            if current_stroke:
                strokes.append(current_stroke)
                current_stroke = []

    if not strokes:
        print("❌ 这是一个空字符！")
        return

    # ==========================================
    # 3. 动态自适应线宽映射 (消灭硬编码常数)
    # ==========================================
    # 找出这个字最快的运笔和最慢的运笔
    min_speed = min(all_speeds) if all_speeds else 0.0
    max_speed = max(all_speeds) if all_speeds else 1.0
    if max_speed == min_speed:
        max_speed += 1e-6 # 防止除以 0

    MAX_WIDTH = 6.0 # 最粗的顿笔
    MIN_WIDTH = 1.0 # 最细的笔锋

    plt.figure(figsize=(6, 6))
    colors = plt.cm.get_cmap('tab10', max(10, len(strokes)))
    
    for i, stroke in enumerate(strokes):
        color = colors(i % 10)
        
        # 顿笔起步小圆点
        plt.plot(stroke[0][0], stroke[0][1], marker='o', markersize=5, color=color)
        
        for j in range(1, len(stroke)):
            x1, y1, _ = stroke[j-1]
            x2, y2, speed = stroke[j]
            
            # --- 核心黑魔法：归一化速度映射 ---
            # 将当前速度压缩到 0 ~ 1 之间 (0表示最慢，1表示最快)
            normalized_speed = (speed - min_speed) / (max_speed - min_speed)
            
            # 速度越快(靠近1)，减去的值越大，线越细；速度越慢(靠近0)，线越粗保持在 MAX_WIDTH
            dynamic_width = MAX_WIDTH - normalized_speed * (MAX_WIDTH - MIN_WIDTH)
            
            plt.plot([x1, x2], [y1, y2], color=color, 
                     linewidth=dynamic_width, 
                     solid_capstyle='round')

    plt.gca().invert_yaxis()  
    plt.axis('equal')
    plt.axis('off')
    plt.title(f"Label: {label}\n(Auto-Scaled Velocity Pressure)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preview_cartesian_dynamic_width()