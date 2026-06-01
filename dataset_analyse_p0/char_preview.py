import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 核心数学拟合引擎 (全新切分逻辑)
# ==========================================
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

def extract_stroke_sequence(stroke_points, prev_end_x, prev_end_y, max_len=15.0):
    """
    【同步升级】动态轨迹切分引擎 (Dynamic Stroke Segmenter)
    """
    if len(stroke_points) < 2: 
        return [], (prev_end_x, prev_end_y)
    
    tokens = []
    curr_start_idx = 0
    
    while curr_start_idx < len(stroke_points) - 1:
        curr_end_idx = curr_start_idx
        chunk_len = 0.0
        chunk_angles = []
        
        # 向前探测，寻找最佳切分点
        for i in range(curr_start_idx + 1, len(stroke_points)):
            x1, y1 = stroke_points[i-1]
            x2, y2 = stroke_points[i]
            dist = math.hypot(x2-x1, y2-y1)
            
            if dist > 1e-4:
                chunk_len += dist
                chunk_angles.append(math.atan2(y2-y1, x2-x1))
                
            # 切分触发条件 1：出现了剧烈的锐角/折线 (大于 60 度)
            angle_diff = 0.0
            if len(chunk_angles) >= 2:
                angle_diff = abs(normalize_angle(chunk_angles[-1] - chunk_angles[-2]))
                
            if angle_diff > math.pi / 3.0: 
                curr_end_idx = i - 1 
                break
                
            # 切分触发条件 2：弧长积攒得足够长了
            if chunk_len > max_len:
                curr_end_idx = i
                break
                
            curr_end_idx = i
        
        # 兜底：如果这截没怎么动，直接跳过
        if chunk_len < 1.0 and curr_end_idx == len(stroke_points) - 1:
            break
            
        chunk_pts = stroke_points[curr_start_idx:max(curr_end_idx+1, curr_start_idx+2)]
        start_x, start_y = chunk_pts[0]
        end_x, end_y = chunk_pts[-1]
        
        dx = start_x - prev_end_x
        dy = start_y - prev_end_y
        
        theta = chunk_angles[0] if chunk_angles else 0.0
        kappa = 0.0
        for i in range(1, len(chunk_angles)):
            kappa += normalize_angle(chunk_angles[i] - chunk_angles[i-1])
            
        tokens.append([dx, dy, chunk_len, theta, kappa, 1.0, 0.0])
        
        prev_end_x, prev_end_y = end_x, end_y
        curr_start_idx = curr_end_idx
        
    return tokens, (prev_end_x, prev_end_y)

# ==========================================
# 2. 原始 txt 解析器
# ==========================================
def parse_omniglot_txt(filepath):
    strokes = []
    current_stroke = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line == "START": continue
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
    return strokes

# ==========================================
# 3. 终极批量照妖镜：N行2列对比阵列
# ==========================================
def compare_batch_raw_vs_parametric(input_dir="strokes_background", num_samples=5):
    if not os.path.exists(input_dir):
        print(f"❌ 找不到文件夹: {input_dir}")
        return

    all_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.txt'):
                all_files.append(os.path.join(root, f))

    if not all_files:
        print("❌ 文件夹里没有找到 txt 文件！")
        return

    sampled_files = random.choices(all_files, k=num_samples)
    print(f"🔍 正在批量验真 {num_samples} 个样本 (已启用动态切分引擎)...")

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples), facecolor='#121212')
    plt.subplots_adjust(wspace=0.1, hspace=0.3, left=0.05, right=0.95, bottom=0.02, top=0.95)
    
    if num_samples == 1:
        axes = np.array([axes])

    for row_idx, filepath in enumerate(sampled_files):
        ax_raw = axes[row_idx, 0]
        ax_recon = axes[row_idx, 1]
        
        ax_raw.set_facecolor('#1E1E20')
        ax_recon.set_facecolor('#1E1E20')

        filename = os.path.basename(filepath)
        raw_strokes = parse_omniglot_txt(filepath)
        
        if not raw_strokes:
            ax_raw.set_title("Empty File", color='red')
            ax_recon.set_title("Empty File", color='red')
            continue

        colors = plt.cm.get_cmap('tab10', max(10, len(raw_strokes)))

        # ==========================================
        # [左列] 纯天然散点连线 (Ground Truth)
        # ==========================================
        for i, stroke in enumerate(raw_strokes):
            xs = [pt[0] for pt in stroke]
            ys = [pt[1] for pt in stroke]
            color = colors(i % 10)
            ax_raw.plot(xs, ys, marker='.', markersize=3, color=color, linewidth=1.5, alpha=0.8)
            ax_raw.plot(xs[0], ys[0], marker='o', markersize=5, color=color)

        ax_raw.invert_yaxis()
        ax_raw.axis('equal')
        ax_raw.axis('off')
        
        if row_idx == 0:
            ax_raw.set_title("Original Point Cloud (Raw)", color='#CCCCCC', fontsize=14, fontweight='bold')
        
        ax_raw.text(-0.1, 0.5, f"Sample {row_idx+1}\n{filename}", transform=ax_raw.transAxes, 
                    color='#888888', fontsize=10, va='center', ha='right')

        # ==========================================
        # [右列] 动态切分提取并积分重绘
        # ==========================================
        prev_end_x, prev_end_y = raw_strokes[0][0][0], raw_strokes[0][0][1]
        
        # 结构改变：为了保持色彩对应，我们将 Token 按原始笔画分组
        parametric_strokes_grouped = [] 
        for stroke in raw_strokes:
            tokens_list, (prev_end_x, prev_end_y) = extract_stroke_sequence(stroke, prev_end_x, prev_end_y)
            parametric_strokes_grouped.append(tokens_list)

        render_prev_x, render_prev_y = raw_strokes[0][0][0], raw_strokes[0][0][1]
        
        # 按原始笔画的索引 (i) 遍历，保证切片出来的子方程拥有相同的父级颜色
        for i, tokens_list in enumerate(parametric_strokes_grouped):
            color = colors(i % 10)
            
            for token in tokens_list:
                dx, dy, L, theta, kappa, W, P = token
                start_x = render_prev_x + dx
                start_y = render_prev_y + dy
                
                steps = 50 
                t_vals = np.linspace(0, 1.0, steps)
                pts_x, pts_y = [], []
                
                for t in t_vals:
                    if abs(kappa) < 1e-4:
                        x = start_x + L * t * math.cos(theta)
                        y = start_y + L * t * math.sin(theta)
                    else:
                        x = start_x + (L / kappa) * (math.sin(theta + kappa * t) - math.sin(theta))
                        y = start_y - (L / kappa) * (math.cos(theta + kappa * t) - math.cos(theta))
                    pts_x.append(x)
                    pts_y.append(y)
                
                for j in range(1, len(pts_x)):
                    x1, y1 = pts_x[j-1], pts_y[j-1]
                    x2, y2 = pts_x[j], pts_y[j]
                    ax_recon.plot([x1, x2], [y1, y2], color=color, linewidth=2.5, solid_capstyle='round')
                    
                # 给每一个被切分出来的 7D Sub-stroke 起点画个点，方便你看算法是在哪里“挥刀切断”的
                ax_recon.plot(pts_x[0], pts_y[0], marker='o', markersize=4, color=color, alpha=0.6)
                
                render_prev_x, render_prev_y = pts_x[-1], pts_y[-1]

        ax_recon.invert_yaxis()
        ax_recon.axis('equal')
        ax_recon.axis('off')
        
        if row_idx == 0:
            ax_recon.set_title("7D Parametric (Chunked)", color='#00FFCC', fontsize=14, fontweight='bold')

    print("🎉 批量验证画廊拼接完毕！正在弹窗...")
    plt.show()

if __name__ == "__main__":
    compare_batch_raw_vs_parametric(input_dir="strokes_background", num_samples=5)